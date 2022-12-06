#pragma once

#include "visited_list_pool.h"
#include "hnswlib.h"
#include <atomic>
#include <random>
#include <stdlib.h>
#include <assert.h>
#include <unordered_set>
#include <list>
#include <iostream>
#include <omp.h>

namespace hnswlib {
    typedef unsigned int tableint; //四个字节，一个tableint存储一个邻居节点的id
    typedef unsigned int linklistsizeint;//其实就是四个字节，对于第0层是size+flag+reserved,对于第0层以上的， 就是size+reserved

    template<typename dist_t>
    class HierarchicalNSW : public AlgorithmInterface<dist_t> {
    public:
        static const tableint max_update_element_locks = 65536;
        HierarchicalNSW(SpaceInterface<dist_t> *s) {
        }

        HierarchicalNSW(SpaceInterface<dist_t> *s, const std::string &location, bool nmslib = false, size_t max_elements=0) {
            loadIndex(location, s, max_elements);
        }

        HierarchicalNSW(SpaceInterface<dist_t> *s, size_t max_elements, size_t M = 16, size_t ef_construction = 200, size_t random_seed = 100) :
                link_list_locks_(max_elements), link_list_update_locks_(max_update_element_locks), element_levels_(max_elements) {
            max_elements_ = max_elements;
            // link_list_locks_ 节点邻居表锁，每个节点一个。
            // link_list_update_locks_  更新占坑锁，它限定了增量的速度，max_update_element_locks决定全局同时最多可以有多少个向量在增加，构建时指定，默认65536
            // 每个节点在哪一层，是vector数组，数组索引是节点内部id
            num_deleted_ = 0;// 删除的节点数目
            data_size_ = s->get_data_size();// 每条原始向量的字节数 dim * sizeof(float)
            fstdistfunc_ = s->get_dist_func();// 度量函数接口
            dist_func_param_ = s->get_dist_func_param();// &dim_
            M_ = M; //每个（非0层）节点可以有多少个邻居, 每层每个元素所能建立的最大连接数
            maxM_ = M_;
            maxM0_ = M_ * 2; //第0层节点可以有多少个邻居 ,第零层一个元素能建立的最大连接数
            ef_construction_ = std::max(ef_construction,M_);//动态候选列表的大小
            ef_ = 10;

            level_generator_.seed(random_seed);// 随机数发生器，决定新增的向量在哪一层
            update_probability_generator_.seed(random_seed + 1);// 随机数发生器，增量更新时，概率的让邻居节点更新邻居。实际上不生效。

            size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint); //第0层邻居表+header的大小,邻居域（每个向量在第0层的近邻向量id）
            size_data_per_element_ = size_links_level0_ + data_size_ + sizeof(labeltype); // 第0层每个元素的大小,向量数据域（原始向量， data_size_），label(向量的业务id),邻居域（每个向量在第0层的近邻向量id）
            offsetData_ = size_links_level0_;// 偏移量，直接定位到第0层每个元素中，原始向量的位置
            label_offset_ = size_links_level0_ + data_size_;// 偏移量，直接定位到第0层每个元素中，label的位置
            offsetLevel0_ = 0;

            data_level0_memory_ = (char *) malloc(max_elements_ * size_data_per_element_); //一次性为第0层申请连续的存储空间, 0层全部数据保存在data_level0_memory_,构造索引时通过参数max_elements指定索引最大向量个数。
            if (data_level0_memory_ == nullptr)
                throw std::runtime_error("Not enough memory");

            cur_element_count = 0;
            // 图操作经常需要判断哪些节点已经走过，这里提供一个已经申请好空间的池子，减少内存频繁申请释放的开销
            visited_list_pool_ = new VisitedListPool(1, max_elements);

            //initializations for special treatment of the first node
            enterpoint_node_ = -1;// 进入点的id
            maxlevel_ = -1;// 最大层数

            //  linkLists_ 一个两个维度上都是变长的二位char型数组，malloc分配相应的内存空间
            //           其中的每一行代表一个节点从第1层到maxLevel层每一层的邻居关系。
            //           每个节点每一层的数据结构主要包括：邻居的数量(size)，保留的两个字节，以及该层的邻居的id
            //           sizeof(void *)与编译器的目标平台有关，就是一个任意类型的指针大小，64位下的值为8
            linkLists_ = (char **) malloc(sizeof(void *) * max_elements_);//存的是每个向量对应列表的地址,二维数组，每一行代表一个节点从第一层到maxlevel层每一层的邻居关系。每个节点每一层的数据结构主要包括：邻居的数量（size),保留的两个字节，以及该层的邻居的id.
            if (linkLists_ == nullptr)
                throw std::runtime_error("Not enough memory: HierarchicalNSW failed to allocate linklists");
            size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint); //每个节点对应邻居跳表，在每一层需要占的空间
            mult_ = 1 / log(1.0 * M_); //用于计算新增向量落在哪一层
            revSize_ = 1.0 / mult_;
        }

        struct CompareByFirst {//这边只是定义了比较的规则，优先队列默认是从大到小排列，因此这边的队列结果还是从大到小排列, //constexpr 使变量获得在编译阶段即可计算出结果的能力
            constexpr bool operator()(std::pair<dist_t, tableint> const &a,
                                      std::pair<dist_t, tableint> const &b) const noexcept {
                return a.first < b.first;//当一个pair a的first值小于另一个pair b的first值时， 称a<b
            }
        };

        ~HierarchicalNSW() {

            free(data_level0_memory_);
            for (tableint i = 0; i < cur_element_count; i++) {
                if (element_levels_[i] > 0)
                    free(linkLists_[i]);
            }
            free(linkLists_); //节点邻居跳表，char**
            delete visited_list_pool_;
        }

        size_t max_elements_;
        size_t cur_element_count;
        size_t size_data_per_element_;//第0层节点表中，需要占多少字节=size_links_level0_ +data_size_ +8
        size_t size_links_per_element_;//每个节点对应邻居跳表，在每一层需要占的空间
        size_t num_deleted_;//有过删除某向量的次数，PS:不考虑增量情况下，它不会发生影响

        size_t M_;
        size_t maxM_;
        size_t maxM0_;
        size_t ef_construction_;

        double mult_, revSize_;
        int maxlevel_;


        VisitedListPool *visited_list_pool_;
        //mutex互斥锁是一个可锁的对象，它被设计成在关键的代码段需要独占访问时发出信号，从而阻止具有相同保护的其他线程并发执行并访问相同的内存位置。
        std::mutex cur_element_count_guard_; //增员减员锁

        std::vector<std::mutex> link_list_locks_;//节点邻居表锁，每个节点一个

        // Locks to prevent race condition during update/insert of an element at same time.
        // Note: Locks for additions can also be used to prevent this race condition if the querying of KNN is not exposed along with update/inserts i.e multithread insert/update/query in parallel.
        std::vector<std::mutex> link_list_update_locks_;//更新占坑锁，它限定了增量的速度。个数由下面的max_update_element_locks决定。
        tableint enterpoint_node_;//随机进度点的内部id, 初始为-1

        size_t size_links_level0_;//第0层节点的邻居表，需要占多少字节
        size_t offsetData_, offsetLevel0_;

        char *data_level0_memory_;
        char **linkLists_; //节点邻居跳表，char*,每个节点对应数据依然是连续数组
        std::vector<int> element_levels_;

        size_t data_size_;

        size_t label_offset_;
        DISTFUNC<dist_t> fstdistfunc_;
        void *dist_func_param_;
        std::unordered_map<labeltype, tableint> label_lookup_;

        std::default_random_engine level_generator_; //随机数发生器，决定新增的向量在哪一层
        std::default_random_engine update_probability_generator_;//随机数发生器，增量更新时，概率的让邻居节点更新邻居，实际上不生效。

        // inline定义的类的内联函数，函数的代码被放入符号表中，在使用时直接进行替换（像宏一样展开），没有了调用的开销，效率也很高。
        // 根据数据id获取label
        inline labeltype getExternalLabel(tableint internal_id) const { //求对外的 index(这边管内部叫idx,外部叫label)
            labeltype return_label;
            memcpy(&return_label,(data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), sizeof(labeltype));
            return return_label;
        }
        // 根据数据id设置label
        inline void setExternalLabel(tableint internal_id, labeltype label) const {//设置外部label的值，用memcpy实现，就是将label的值写到前面的内存里面去
            memcpy((data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), &label, sizeof(labeltype));
        }
        // 根据id返回label的指针
        inline labeltype *getExternalLabeLp(tableint internal_id) const {//获取外部的label的指针
            return (labeltype *) (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_);
        }
        // 根据id返回原始向量的指针
        inline char *getDataByInternalId(tableint internal_id) const {//通过内部的index值获取data
            return (data_level0_memory_ + internal_id * size_data_per_element_ + offsetData_);
        }
        // 获取随机层数
        int getRandomLevel(double reverse_size) {
            std::uniform_real_distribution<double> distribution(0.0, 1.0);
            double r = -log(distribution(level_generator_)) * reverse_size;
            return (int) r;
        }

        // 算法2：搜索第layer层中离目标最近的ef个邻居
        //searchBaseLayer在addPoint和updatePoint的时候调用，也就是在建图的时候调用，可以计算各种层，不仅仅是底层
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
        searchBaseLayer(tableint ep_id, const void *data_point, int layer) {//在某一层search，参数需要直到是哪层，enter_point, data_point是query的向量
            // data_point 目标点
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();// VisitedList存储已访问过的节点，下面进行其初始化过程
            vl_type *visited_array = vl->mass; // 新建vl_type实例
            vl_type visited_array_tag = vl->curV; // visited_array_tag初始化为-1

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;//列表W，存储最终的ef个元素// top_candidates 结果集W
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidateSet;//列表C，存储候选元素// candidateSet 动态候选集C

            dist_t lowerBound;//存储W中距离Q的最远距离
            if (!isMarkedDeleted(ep_id)) {// 如果当前节点没有被标记为删除
                dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);//先从enter_point开始计算，计算dist，然后放入结果中//计算当前节点到目标节点的距离，记为dist
                top_candidates.emplace(dist, ep_id);// 将当前节点插入结果集W中
                lowerBound = dist; //更新结果集W中的最远距离为dist, 存储W中距离Q的最远距离， lowerBound =dist -distance(ep_id, q)
                candidateSet.emplace(-dist, ep_id);
            } else {// 如果当前节点已经被标记为删除（注意删除的节点不插入结果集W，只插入动态候选集C）
                lowerBound = std::numeric_limits<dist_t>::max();// 更新结果集W中的最远距离为当前数据类型dist_t的最大值
                candidateSet.emplace(-lowerBound, ep_id);// 将该节点插入动态候选集C中
            }


            // 对第0层执行算法2，获得结果集W
            /*
            visited_array   记录已经遍历过的节点
            candidateSet    动态候选集，相当于论文中的C
            top_candidates  结果集，相当于论文中的W
            */
            visited_array[ep_id] = visited_array_tag;//这个相当于列表V，存储计算过距离的元素

            while (!candidateSet.empty()) {//如果候选集C不为空，从C中取出距离q最近的元素curNode,// 循环的停止条件是动态候选集C为空，前面已经把enterpoint添加到动态候选集中了
                std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();//如果distance(currNode, q)>lowerBound那么就返回W，否则获取currNode的所有邻居neighbors// 弹出动态候选集C中当前最近的节点，记为curr_el_pair
                if ((-curr_el_pair.first) > lowerBound && top_candidates.size() == ef_construction_) {// 如果当前节点到检索目标的距离大于结果集W中的最大距离，且结果集W的大小|W|已等于ef
                    break;// 直接结束循环（接下来进入算法的最后阶段，即释放访问列表存储空间，算法结束，返回结果集W）
                }
                candidateSet.pop();// 将该节点从动态候选集C中删除

                tableint curNodeNum = curr_el_pair.second;//获取node idx

                std::unique_lock <std::mutex> lock(link_list_locks_[curNodeNum]);

                int *data;// = (int *)(linkList0_ + curNodeNum * size_links_per_element0_);
                // data指向的是当前节点的首地址
                if (layer == 0) {
                    data = (int*)get_linklist0(curNodeNum);
                } else {
                    data = (int*)get_linklist(curNodeNum, layer);
//                    data = (int *) (linkLists_[curNodeNum] + (layer - 1) * size_links_per_element_);
                }
                size_t size = getListCount((linklistsizeint*)data);//邻居的数量// size为当前节点的邻居数目
                tableint *datal = (tableint *) (data + 1); //前面是L, 后面是一，取出第一个邻居// +1 即跳转4个字节，刚好跳过header，指向邻居列表
#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);//SSE使用_mm_prefetch加速计算，可以在实际当前运算与数据从内存到cache的加载并行，从而达到加速的目的
                _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
#endif
                // 遍历当前节点的邻居列表
                for (size_t j = 0; j < size; j++) {//遍历所有的邻居
                    tableint candidate_id = *(datal + j); //由这个得到currObj1(getDataByInternalId)
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                    _mm_prefetch((char *) (visited_array + *(datal + j + 1)), _MM_HINT_T0);
                    _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
#endif
                    if (visited_array[candidate_id] == visited_array_tag) continue;// 如果该邻居已经被遍历过，就跳过该邻居
                    visited_array[candidate_id] = visited_array_tag;//标记为visited,加入列表V // 如果该邻居没有被遍历过，就标记该邻居为已遍历
                    char *currObj1 = (getDataByInternalId(candidate_id));

                    dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);//计算邻居和query的距离// 计算该邻居到目标的距离，记为dist1
                    // 如果结果集W的大小|W|小于ef，或者该点距离dist1小于W中的最大值lowerBound
                    if (top_candidates.size() < ef_construction_ || lowerBound > dist1) {//dist1<W中距离q的最远距离或者W.size<ef, neighbors[i]加入列表C, W
                        candidateSet.emplace(-dist1, candidate_id); // 将该节点插入动态候选集C
#ifdef USE_SSE
                        _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
#endif

                        if (!isMarkedDeleted(candidate_id))// 如果该节点没有被标记为删除
                            top_candidates.emplace(dist1, candidate_id); // 就将其插入结果集W
                        // 如果结果集W大小超过ef
                        if (top_candidates.size() > ef_construction_)//W.size>ef,取出W中距离q最远的元素
                            top_candidates.pop();// 就把结果集W中距离最大的节点删除
                        // 如果结果集W不为空 
                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;//更新loweBound=W中距离q最远的元素// 更新当前结果集W中离目标最远的距离
                    }
                }
            }
            visited_list_pool_->releaseVisitedList(vl);// 释放存储空间

            return top_candidates;// 返回结果集W
        }

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
        searchLayerL1(tableint ep_id, const void *data_point, int thread_num) const{//在某一层search，参数需要直到是哪层，enter_point, data_point是query的向量
            // data_point 目标点
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();// VisitedList存储已访问过的节点，下面进行其初始化过程
            vl_type *visited_array = vl->mass; // 新建vl_type实例
            vl_type visited_array_tag = vl->curV; // visited_array_tag初始化为-1

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;//列表W，存储最终的ef个元素// top_candidates 结果集W
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidateSet;//列表C，存储候选元素// candidateSet 动态候选集C

            dist_t lowerBound;//存储W中距离Q的最远距离
            if (!isMarkedDeleted(ep_id)) {// 如果当前节点没有被标记为删除
                dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);//先从enter_point开始计算，计算dist，然后放入结果中//计算当前节点到目标节点的距离，记为dist
                top_candidates.emplace(dist, ep_id);// 将当前节点插入结果集W中
                lowerBound = dist; //更新结果集W中的最远距离为dist, 存储W中距离Q的最远距离， lowerBound =dist -distance(ep_id, q)
                candidateSet.emplace(-dist, ep_id);
            } else {// 如果当前节点已经被标记为删除（注意删除的节点不插入结果集W，只插入动态候选集C）
                lowerBound = std::numeric_limits<dist_t>::max();// 更新结果集W中的最远距离为当前数据类型dist_t的最大值
                candidateSet.emplace(-lowerBound, ep_id);// 将该节点插入动态候选集C中
            }


            // 对第0层执行算法2，获得结果集W
            /*
            visited_array   记录已经遍历过的节点
            candidateSet    动态候选集，相当于论文中的C
            top_candidates  结果集，相当于论文中的W
            */
            visited_array[ep_id] = visited_array_tag;//这个相当于列表V，存储计算过距离的元素

            while (!candidateSet.empty()) {//如果候选集C不为空，从C中取出距离q最近的元素curNode,// 循环的停止条件是动态候选集C为空，前面已经把enterpoint添加到动态候选集中了
                std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();//如果distance(currNode, q)>lowerBound那么就返回W，否则获取currNode的所有邻居neighbors// 弹出动态候选集C中当前最近的节点，记为curr_el_pair
                if ((-curr_el_pair.first) > lowerBound && top_candidates.size() == thread_num) {// 如果当前节点到检索目标的距离大于结果集W中的最大距离，且结果集W的大小|W|已等于ef
                    break;// 直接结束循环（接下来进入算法的最后阶段，即释放访问列表存储空间，算法结束，返回结果集W）
                }
                candidateSet.pop();// 将该节点从动态候选集C中删除

                tableint curNodeNum = curr_el_pair.second;//获取node idx

                // std::unique_lock <std::mutex> lock(link_list_locks_[curNodeNum]);

                int *data;// = (int *)(linkList0_ + curNodeNum * size_links_per_element0_);
                // data指向的是当前节点的首地址

                data = (int*)get_linklist(curNodeNum, 1);

                size_t size = getListCount((linklistsizeint*)data);//邻居的数量// size为当前节点的邻居数目
                tableint *datal = (tableint *) (data + 1); //前面是L, 后面是一，取出第一个邻居// +1 即跳转4个字节，刚好跳过header，指向邻居列表
#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);//SSE使用_mm_prefetch加速计算，可以在实际当前运算与数据从内存到cache的加载并行，从而达到加速的目的
                _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
#endif
                // 遍历当前节点的邻居列表
                for (size_t j = 0; j < size; j++) {//遍历所有的邻居
                    tableint candidate_id = *(datal + j); //由这个得到currObj1(getDataByInternalId)
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                    _mm_prefetch((char *) (visited_array + *(datal + j + 1)), _MM_HINT_T0);
                    _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
#endif
                    if (visited_array[candidate_id] == visited_array_tag) continue;// 如果该邻居已经被遍历过，就跳过该邻居
                    visited_array[candidate_id] = visited_array_tag;//标记为visited,加入列表V // 如果该邻居没有被遍历过，就标记该邻居为已遍历
                    char *currObj1 = (getDataByInternalId(candidate_id));

                    dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);//计算邻居和query的距离// 计算该邻居到目标的距离，记为dist1
                    // 如果结果集W的大小|W|小于ef，或者该点距离dist1小于W中的最大值lowerBound
                    if (top_candidates.size() < thread_num || lowerBound > dist1) {//dist1<W中距离q的最远距离或者W.size<ef, neighbors[i]加入列表C, W
                        candidateSet.emplace(-dist1, candidate_id); // 将该节点插入动态候选集C
#ifdef USE_SSE
                        _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
#endif

                        if (!isMarkedDeleted(candidate_id))// 如果该节点没有被标记为删除
                            top_candidates.emplace(dist1, candidate_id); // 就将其插入结果集W
                        // 如果结果集W大小超过ef
                        if (top_candidates.size() > thread_num)//W.size>ef,取出W中距离q最远的元素
                            top_candidates.pop();// 就把结果集W中距离最大的节点删除
                        // 如果结果集W不为空 
                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;//更新loweBound=W中距离q最远的元素// 更新当前结果集W中离目标最远的距离
                    }
                }
            }
            visited_list_pool_->releaseVisitedList(vl);// 释放存储空间

            return top_candidates;// 返回结果集W
        }




        mutable std::atomic<long> metric_distance_computations;
        mutable std::atomic<long> metric_hops;//可以使用任意的类型作为模板参数。在多线程中如果使用了原子变量，
                                              //其本身就保证了数据访问的互斥性，所以不需要使用互斥量来保护该变量了
        
        //维护一个长度不大于ef_construction的动态list，记为W。每次从动态list中取最近的点，遍历它的邻居节点，
        //如果它的邻居没有被遍历过visited，那么当结果集小于ef_construction，或者该节点比结果集中最远的点离目标近时，则把它添加到W中，
        //如果该点没有被标记为删除，则添加到结果集。如果添加后结果集数量多于ef_construction，则把最远的pop出来 

        //searchBaseLayerST只计算底层，searchBaseLayer可以计算各种层，包括底层，searchBaseLayerST是搜索的时候调用，只计算底层
        template <bool has_deletions, bool collect_metrics=false>
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
        searchBaseLayerST(tableint ep_id, const void *data_point, size_t ef) const {//只计算底层，传入的参数是ef
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;//当前值为-1

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;//W 顶部是距离最大的元素，用于删和返回
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;//C 顶部最小，用于提取元素

            dist_t lowerBound;
            //num_deleted_统计flag=1的元素数，删除和 被访问不同
            if (!has_deletions || !isMarkedDeleted(ep_id)) {//currObj没被删时或num_deleted_为0时进入
                dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);//q和data的距离
                lowerBound = dist;
                top_candidates.emplace(dist, ep_id);
                candidate_set.emplace(-dist, ep_id);
            } else {
                lowerBound = std::numeric_limits<dist_t>::max();
                candidate_set.emplace(-lowerBound, ep_id);//倒序排列
            }
            //这个时候candidate_set和top_candidates只有一个值

            visited_array[ep_id] = visited_array_tag;//存为1，表示已被访问

            while (!candidate_set.empty()) {

                std::pair<dist_t, tableint> current_node_pair = candidate_set.top();

                if ((-current_node_pair.first) > lowerBound && (top_candidates.size() == ef || has_deletions == false)) {
                    break;
                }
                candidate_set.pop();

                tableint current_node_id = current_node_pair.second;
                int *data = (int *) get_linklist0(current_node_id);
                size_t size = getListCount((linklistsizeint*)data);
//                bool cur_node_deleted = isMarkedDeleted(current_node_id);
                if(collect_metrics){
                    metric_hops++;
                    metric_distance_computations+=size;
                }

#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
                _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
                _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
                _mm_prefetch((char *) (data + 2), _MM_HINT_T0);
#endif

                for (size_t j = 1; j <= size; j++) {
                    int candidate_id = *(data + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                    _mm_prefetch((char *) (visited_array + *(data + j + 1)), _MM_HINT_T0);
                    _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_,
                                 _MM_HINT_T0);////////////
#endif
                    if (!(visited_array[candidate_id] == visited_array_tag)) {

                        visited_array[candidate_id] = visited_array_tag;

                        char *currObj1 = (getDataByInternalId(candidate_id));
                        dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);

                        if (top_candidates.size() < ef || lowerBound > dist) {
                            candidate_set.emplace(-dist, candidate_id);
#ifdef USE_SSE
                            _mm_prefetch(data_level0_memory_ + candidate_set.top().second * size_data_per_element_ +
                                         offsetLevel0_,///////////
                                         _MM_HINT_T0);////////////////////////
#endif

                            if (!has_deletions || !isMarkedDeleted(candidate_id))
                                top_candidates.emplace(dist, candidate_id);

                            if (top_candidates.size() > ef)
                                top_candidates.pop();

                            if (!top_candidates.empty())
                                lowerBound = top_candidates.top().first;
                        }
                    }
                }
            }

            visited_list_pool_->releaseVisitedList(vl);
            return top_candidates;//返回动态列表，也就是返回layer层中距离q最近的ef个邻居
        }

//         //searchBaseLayerST只计算底层，searchBaseLayer可以计算各种层，包括底层，searchBaseLayerST是搜索的时候调用，只计算底层
//         template <bool has_deletions, bool collect_metrics=false>
//         std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
//         parallelsearchBaseLayerST(tableint ep_id, const void *data_point, size_t ef, size_t thread_num) const {//只计算底层，传入的参数是ef
//             VisitedList *vl = visited_list_pool_->getFreeVisitedList();
//             vl_type *visited_array = vl->mass;
//             vl_type visited_array_tag = vl->curV;//当前值为-1

//             std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;//W 顶部是距离最大的元素，用于删和返回
//             std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;//C 顶部最小，用于提取元素

//             dist_t lowerBound;
//             //num_deleted_统计flag=1的元素数，删除和 被访问不同
//             if (!has_deletions || !isMarkedDeleted(ep_id)) {//currObj没被删时或num_deleted_为0时进入
//                 dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);//q和data的距离
//                 lowerBound = dist;
//                 top_candidates.emplace(dist, ep_id);
//                 candidate_set.emplace(-dist, ep_id);
//             } else {
//                 lowerBound = std::numeric_limits<dist_t>::max();
//                 candidate_set.emplace(-lowerBound, ep_id);//倒序排列
//             }
//             //这个时候candidate_set和top_candidates只有一个值

//             visited_array[ep_id] = visited_array_tag;//存为1，表示已被访问

//             while (!candidate_set.empty()) {

//                 std::pair<dist_t, tableint> current_node_pair = candidate_set.top();

//                 if ((-current_node_pair.first) > lowerBound && (top_candidates.size() == ef || has_deletions == false)) {
//                     break;
//                 }
//                 candidate_set.pop();

//                 tableint current_node_id = current_node_pair.second;
//                 int *data = (int *) get_linklist0(current_node_id);
//                 size_t size = getListCount((linklistsizeint*)data);
// //                bool cur_node_deleted = isMarkedDeleted(current_node_id);
//                 if(collect_metrics){
//                     metric_hops++;
//                     metric_distance_computations+=size;
//                 }

// #ifdef USE_SSE
//                 _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
//                 _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
//                 _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
//                 _mm_prefetch((char *) (data + 2), _MM_HINT_T0);
// #endif

//                 for (size_t j = 1; j <= size; j++) {
//                     int candidate_id = *(data + j);
// //                    if (candidate_id == 0) continue;
// #ifdef USE_SSE
//                     _mm_prefetch((char *) (visited_array + *(data + j + 1)), _MM_HINT_T0);
//                     _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_,
//                                  _MM_HINT_T0);////////////
// #endif
//                     if (!(visited_array[candidate_id] == visited_array_tag)) {

//                         visited_array[candidate_id] = visited_array_tag;

//                         char *currObj1 = (getDataByInternalId(candidate_id));
//                         dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);

//                         if (top_candidates.size() < ef || lowerBound > dist) {
//                             candidate_set.emplace(-dist, candidate_id);
// #ifdef USE_SSE
//                             _mm_prefetch(data_level0_memory_ + candidate_set.top().second * size_data_per_element_ +
//                                          offsetLevel0_,///////////
//                                          _MM_HINT_T0);////////////////////////
// #endif

//                             if (!has_deletions || !isMarkedDeleted(candidate_id))
//                                 top_candidates.emplace(dist, candidate_id);

//                             if (top_candidates.size() > ef)
//                                 top_candidates.pop();

//                             if (!top_candidates.empty())
//                                 lowerBound = top_candidates.top().first;
//                         }
//                     }
//                 }
//             }

//             visited_list_pool_->releaseVisitedList(vl);
//             return top_candidates;//返回动态列表，也就是返回layer层中距离q最近的ef个邻居
//         }

       //算法4 启发式方法选择邻居，从top-candidates中选择距离q最近的M个元素
        void getNeighborsByHeuristic2(
                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        const size_t M) {
            //如果top-candidates中元素个数小于M，直接return
            if (top_candidates.size() < M) {
                return;
            }
            //queue_closest是working queue for the candidates,论文中是W，存放候选者
            std::priority_queue<std::pair<dist_t, tableint>> queue_closest;
            //return_list存放最终的M个结果，论文中是R，初始为空集
            std::vector<std::pair<dist_t, tableint>> return_list;
            //将queue_closest初始化为top_candidates,论文中为WC
            while (top_candidates.size() > 0) {
                queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
                top_candidates.pop();
            }
            //当queue_closest中的元素大于0
            while (queue_closest.size()) {
                //如果return_list中的元素个数已经大于等于M，那么启发式查找过程结束
                if (return_list.size() >= M)
                    break;
                std::pair<dist_t, tableint> curent_pair = queue_closest.top();//current_pair是queue_closest(W)的元素
                dist_t dist_to_query = -curent_pair.first;//dist_to_query是current_pair与query的距离
                queue_closest.pop();//queue_closest元素减一
                bool good = true;

                //对于return_list(R)中的每一个元素
                for (std::pair<dist_t, tableint> second_pair : return_list) {
                    dist_t curdist =
                            fstdistfunc_(getDataByInternalId(second_pair.second), //curdist是curent_pair与return_list(R)中的每个元素的距离
                                         getDataByInternalId(curent_pair.second),
                                         dist_func_param_);;
                    //如果curent_pair与已经与q连接元素的距离小于curent_pair与query的距离
                    if (curdist < dist_to_query) {
                        //current_pair将不会作为q的邻居返回
                        good = false;
                        break;
                    }
                }
                //反之，见论文中Fig.2
                //那么将curent_pair并入return_list(即论文中的R)
                if (good) {
                    return_list.push_back(curent_pair);
                }
            }

            for (std::pair<dist_t, tableint> curent_pair : return_list) {
                top_candidates.emplace(-curent_pair.first, curent_pair.second);
            }
        }

        //返回一个值在data_level0_memory_结构下的。。指针，offsetLevel0_是啥啊（每个内存格是4个字节）
        linklistsizeint *get_linklist0(tableint internal_id) const {
            return (linklistsizeint *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
        };
        //差别，重载了，参数量不同
        linklistsizeint *get_linklist0(tableint internal_id, char *data_level0_memory_) const {
            return (linklistsizeint *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
        };
        //返回一个值在linkLists_结构下的指向其某一层的指针
        linklistsizeint *get_linklist(tableint internal_id, int level) const {
            return (linklistsizeint *) (linkLists_[internal_id] + (level - 1) * size_links_per_element_);
        };

        linklistsizeint *get_linklist_at_level(tableint internal_id, int level) const {
            return level == 0 ? get_linklist0(internal_id) : get_linklist(internal_id, level);
        };
        //每一层添加neighbors与q(data_point)的连接，相互连接新元素（双向）, data_point是这个点的data, tabelint cur_c是这个点的内部label，函数里面只用到了cur_c
        tableint mutuallyConnectNewElement(const void *data_point, tableint cur_c,
                                       std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        int level, bool isUpdate) {
            size_t Mcurmax = level ? maxM_ : maxM0_;//获取该层的最大连接数，非0层maxM_,第0层maxM0_
            getNeighborsByHeuristic2(top_candidates, M_); //启发式算法获取top_candidates
            if (top_candidates.size() > M_)
                throw std::runtime_error("Should be not be more than M_ candidates returned by the heuristic");

            std::vector<tableint> selectedNeighbors;
            selectedNeighbors.reserve(M_); //选M_个邻居
            while (top_candidates.size() > 0) {
                selectedNeighbors.push_back(top_candidates.top().second);
                top_candidates.pop();
            }

            tableint next_closest_entry_point = selectedNeighbors.back();

            {
                linklistsizeint *ll_cur;
                if (level == 0)
                    ll_cur = get_linklist0(cur_c);//获取cur_c的邻居
                else
                    ll_cur = get_linklist(cur_c, level);//获取cur_c的邻居，linkLists_是节点邻居跳表

                if (*ll_cur && !isUpdate) {
                    throw std::runtime_error("The newly inserted element should have blank link list");
                }
                setListCount(ll_cur,selectedNeighbors.size());//ll_cur指向邻居表
                tableint *data = (tableint *) (ll_cur + 1);//cur_c的邻居
                for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {//遍历cur_c的邻居的大小
                    if (data[idx] && !isUpdate)
                        throw std::runtime_error("Possible memory corruption");
                    if (level > element_levels_[selectedNeighbors[idx]])
                        throw std::runtime_error("Trying to make a link on a non-existent level");

                    data[idx] = selectedNeighbors[idx]; //连接selectedNeighbors[idx]和cur_c, 之前的邻居节点是0~idx-1

                }
            }
            //更新邻居的邻居
            for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {

                std::unique_lock <std::mutex> lock(link_list_locks_[selectedNeighbors[idx]]);

                linklistsizeint *ll_other;
                if (level == 0)
                    ll_other = get_linklist0(selectedNeighbors[idx]);
                else
                    ll_other = get_linklist(selectedNeighbors[idx], level);

                size_t sz_link_list_other = getListCount(ll_other);//邻居的数量

                if (sz_link_list_other > Mcurmax)
                    throw std::runtime_error("Bad value of sz_link_list_other");
                if (selectedNeighbors[idx] == cur_c)
                    throw std::runtime_error("Trying to connect an element to itself");
                if (level > element_levels_[selectedNeighbors[idx]])
                    throw std::runtime_error("Trying to make a link on a non-existent level");

                tableint *data = (tableint *) (ll_other + 1);

                bool is_cur_c_present = false;
                if (isUpdate) {
                    for (size_t j = 0; j < sz_link_list_other; j++) {
                        if (data[j] == cur_c) {//cur_c已经是邻居的节点里面了，也就是边已经连接好了,data[j]存储的是需要处理的节点的label
                            is_cur_c_present = true;
                            break;
                        }
                    }
                }

                // If cur_c is already present in the neighboring connections of `selectedNeighbors[idx]` then no need to modify any connections or run the heuristics.
                if (!is_cur_c_present) {//如果这个节点没有处理
                    if (sz_link_list_other < Mcurmax) {//邻居数量少于最大连接边数量
                        data[sz_link_list_other] = cur_c;//cur_c和这个邻居连接！！！！！！！！！！！！！！！
                        setListCount(ll_other, sz_link_list_other + 1);//ll_other邻居表的数量+1
                    } else {
                        // finding the "weakest" element to replace it with the new one
                        //cur_c和selectedNeighbors[idx]的距离
                        dist_t d_max = fstdistfunc_(getDataByInternalId(cur_c), getDataByInternalId(selectedNeighbors[idx]),
                                                    dist_func_param_);
                        // Heuristic:
                        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                        candidates.emplace(d_max, cur_c);//优先队列加入cur_c

                        for (size_t j = 0; j < sz_link_list_other; j++) {//遍历邻居点，candidates里面是各个邻居点和selectedNeighbors[idx]的距离
                            candidates.emplace(
                                    fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(selectedNeighbors[idx]),
                                                 dist_func_param_), data[j]);
                        }

                        getNeighborsByHeuristic2(candidates, Mcurmax);//启发式算法裁边

                        int indx = 0;
                        while (candidates.size() > 0) {
                            data[indx] = candidates.top().second;//根据裁边的结果，连接！！！！！，data里面存储的是selectedNeighbors[idx]的邻居
                            candidates.pop();
                            indx++;
                        }

                        setListCount(ll_other, indx);//设定selectedNeighbors[idx]的邻居表的大小
                        // Nearest K:
                        /*int indx = -1;
                        for (int j = 0; j < sz_link_list_other; j++) {
                            dist_t d = fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(rez[idx]), dist_func_param_);
                            if (d > d_max) {
                                indx = j;
                                d_max = d;
                            }
                        }
                        if (indx >= 0) {
                            data[indx] = cur_c;
                        } */
                    }
                }
            }

            return next_closest_entry_point;
        }

        std::mutex global;
        size_t ef_;

        void setEf(size_t ef) {
            ef_ = ef;
        }


        std::priority_queue<std::pair<dist_t, tableint>> searchKnnInternal(void *query_data, int k) {
            std::priority_queue<std::pair<dist_t, tableint  >> top_candidates;
            if (cur_element_count == 0) return top_candidates;
            tableint currObj = enterpoint_node_;//当前层的enterpoint
            dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);//curdist 当前enterpoint到检索目标的距离
            //从最大层开始遍历，从maxlevel到1层，找到离目标最近的一个元素的id，并赋值给currobj
            for (size_t level = maxlevel_; level > 0; level--) {
                bool changed = true;
                while (changed) {
                    changed = false;//首先没有变化，表示在同一层中搜索
                    int *data;
                    data = (int *) get_linklist(currObj,level);//获取currObj的邻居
                    int size = getListCount(data);//size:该节点的邻居数目
                    tableint *datal = (tableint *) (data + 1);//指向了data第一个邻居的id
                    for (int i = 0; i < size; i++) {//遍历该节点的所有邻居，对于currObj的每一个邻居，计算它与 query_data的距离，并及时更新currObj与curdist
                        tableint cand = datal[i];//邻居的id
                        if (cand < 0 || cand > max_elements_)
                            throw std::runtime_error("cand error");
                        dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);//根据id获取邻居并计算其到query data的距离

                        if (d < curdist) {//如果这个邻居与query的距离比curdist还要小，更新curdist为这个邻居，changed改为true.
                            curdist = d; //更新当前enterpoint到检索目标的距离
                            currObj = cand;//更新enterpoint
                            changed = true;//更新enterpoint更改标记为true,再次进入循环
                        }
                    }
                }
            }

            if (num_deleted_) {
                std::priority_queue<std::pair<dist_t, tableint  >> top_candidates1=searchBaseLayerST<true>(currObj, query_data,
                                                                                                           ef_);
                top_candidates.swap(top_candidates1);
            }
            else{
                std::priority_queue<std::pair<dist_t, tableint  >> top_candidates1=searchBaseLayerST<false>(currObj, query_data,
                                                                                                            ef_);
                top_candidates.swap(top_candidates1);
            }

            while (top_candidates.size() > k) {
                top_candidates.pop();
            }
            return top_candidates;
        };

        void resizeIndex(size_t new_max_elements){//有必要的话缩减索引
            if (new_max_elements<cur_element_count)
                throw std::runtime_error("Cannot resize, max element is less than the current number of elements");


            delete visited_list_pool_;
            visited_list_pool_ = new VisitedListPool(1, new_max_elements);


            element_levels_.resize(new_max_elements);

            std::vector<std::mutex>(new_max_elements).swap(link_list_locks_);

            // Reallocate base layer
            char * data_level0_memory_new = (char *) realloc(data_level0_memory_, new_max_elements * size_data_per_element_);
            if (data_level0_memory_new == nullptr)
                throw std::runtime_error("Not enough memory: resizeIndex failed to allocate base layer");
            data_level0_memory_ = data_level0_memory_new;

            // Reallocate all other layers
            char ** linkLists_new = (char **) realloc(linkLists_, sizeof(void *) * new_max_elements);
            if (linkLists_new == nullptr)
                throw std::runtime_error("Not enough memory: resizeIndex failed to allocate other layers");
            linkLists_ = linkLists_new;

            max_elements_ = new_max_elements;
        }

        void saveIndex(const std::string &location) {
            std::ofstream output(location, std::ios::binary);
            std::streampos position;

            writeBinaryPOD(output, offsetLevel0_);
            writeBinaryPOD(output, max_elements_);
            writeBinaryPOD(output, cur_element_count);
            writeBinaryPOD(output, size_data_per_element_);
            writeBinaryPOD(output, label_offset_);
            writeBinaryPOD(output, offsetData_);
            writeBinaryPOD(output, maxlevel_);
            writeBinaryPOD(output, enterpoint_node_);
            writeBinaryPOD(output, maxM_);

            writeBinaryPOD(output, maxM0_);
            writeBinaryPOD(output, M_);
            writeBinaryPOD(output, mult_);
            writeBinaryPOD(output, ef_construction_);

            output.write(data_level0_memory_, cur_element_count * size_data_per_element_);
            //print(linkLists)
            std::ofstream dataFile;
            dataFile.open("./linkLists.txt", std::ios::app);
            if(!dataFile.is_open()){
                std::cout<<"file open failed"<<std::endl;
                return;
            }
            for (size_t i = 0; i < cur_element_count; i++) {
                unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
                // writeBinaryPOD(output, linkListSize);
                // if (linkListSize)
                //     output.write(linkLists_[i], linkListSize);
                dataFile << "cue_element_count: "<<i <<std::endl;
                dataFile << linkListSize<<std::endl;
            }

            //print degree of each node in level 0
            std::ofstream dataFile_level0;
            dataFile_level0.open("./linkLists_level0.txt", std::ios::app);
            if(!dataFile_level0.is_open()){
                std::cout<<"file open failed"<<std::endl;
                return;
            }
            for (size_t i = 0; i < cur_element_count; i++) {
                // linklistsizeint* linkListSize = get_linklist0(cur_element_count);
                // int size = getListCount(linkListSize);

                int *data = (int *) get_linklist0(i);
                size_t size = getListCount((linklistsizeint*)data);

                dataFile_level0 << "cue_element_count: "<<i <<std::endl;
                dataFile_level0 << size<<std::endl;
            }

            for (size_t i = 0; i < cur_element_count; i++) {
                unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
                writeBinaryPOD(output, linkListSize);
                if (linkListSize)
                    output.write(linkLists_[i], linkListSize);
            }
            output.close();
        }

        void loadIndex(const std::string &location, SpaceInterface<dist_t> *s, size_t max_elements_i=0) {
            std::ifstream input(location, std::ios::binary);

            if (!input.is_open())
                throw std::runtime_error("Cannot open file");

            // get file size:
            input.seekg(0,input.end);
            std::streampos total_filesize=input.tellg();
            input.seekg(0,input.beg);

            readBinaryPOD(input, offsetLevel0_);
            readBinaryPOD(input, max_elements_);
            readBinaryPOD(input, cur_element_count);

            size_t max_elements = max_elements_i;
            if(max_elements < cur_element_count)
                max_elements = max_elements_;
            max_elements_ = max_elements;
            readBinaryPOD(input, size_data_per_element_);
            readBinaryPOD(input, label_offset_);
            readBinaryPOD(input, offsetData_);
            readBinaryPOD(input, maxlevel_);
            readBinaryPOD(input, enterpoint_node_);

            readBinaryPOD(input, maxM_);
            readBinaryPOD(input, maxM0_);
            readBinaryPOD(input, M_);
            readBinaryPOD(input, mult_);
            readBinaryPOD(input, ef_construction_);


            data_size_ = s->get_data_size();
            fstdistfunc_ = s->get_dist_func(); //距离计算
            dist_func_param_ = s->get_dist_func_param();  //get dim

            auto pos=input.tellg();


            /// Optional - check if index is ok:

            input.seekg(cur_element_count * size_data_per_element_,input.cur);
            for (size_t i = 0; i < cur_element_count; i++) {
                if(input.tellg() < 0 || input.tellg()>=total_filesize){
                    throw std::runtime_error("Index seems to be corrupted or unsupported");
                }

                unsigned int linkListSize;
                readBinaryPOD(input, linkListSize);
                if (linkListSize != 0) {
                    input.seekg(linkListSize,input.cur);
                }
            }

            // throw exception if it either corrupted or old index
            if(input.tellg()!=total_filesize)
                throw std::runtime_error("Index seems to be corrupted or unsupported");

            input.clear();

            /// Optional check end

            input.seekg(pos,input.beg);

            data_level0_memory_ = (char *) malloc(max_elements * size_data_per_element_);
            if (data_level0_memory_ == nullptr)
                throw std::runtime_error("Not enough memory: loadIndex failed to allocate level0");
            input.read(data_level0_memory_, cur_element_count * size_data_per_element_);

            size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);

            size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
            std::vector<std::mutex>(max_elements).swap(link_list_locks_);
            std::vector<std::mutex>(max_update_element_locks).swap(link_list_update_locks_);

            visited_list_pool_ = new VisitedListPool(1, max_elements);

            linkLists_ = (char **) malloc(sizeof(void *) * max_elements);
            if (linkLists_ == nullptr)
                throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklists");
            element_levels_ = std::vector<int>(max_elements);
            revSize_ = 1.0 / mult_;
            ef_ = 10;
            for (size_t i = 0; i < cur_element_count; i++) {
                label_lookup_[getExternalLabel(i)]=i;
                unsigned int linkListSize;
                readBinaryPOD(input, linkListSize);
                if (linkListSize == 0) {
                    element_levels_[i] = 0;

                    linkLists_[i] = nullptr;
                } else {
                    element_levels_[i] = linkListSize / size_links_per_element_;
                    linkLists_[i] = (char *) malloc(linkListSize);
                    if (linkLists_[i] == nullptr)
                        throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklist");
                    input.read(linkLists_[i], linkListSize);
                }
            }

            for (size_t i = 0; i < cur_element_count; i++) {
                if(isMarkedDeleted(i))
                    num_deleted_ += 1;
            }

            input.close();

            return;
        }

        template<typename data_t>
        std::vector<data_t> getDataByLabel(labeltype label) const  //bindings.cpp里面调用
        {
            tableint label_c;
            auto search = label_lookup_.find(label);
            if (search == label_lookup_.end() || isMarkedDeleted(search->second)) {
                throw std::runtime_error("Label not found");
            }
            label_c = search->second;//内部的index

            char* data_ptrv = getDataByInternalId(label_c);//根据内部id得到data
            size_t dim = *((size_t *) dist_func_param_);//维度
            std::vector<data_t> data;
            data_t* data_ptr = (data_t*) data_ptrv;
            for (int i = 0; i < dim; i++) {
                data.push_back(*data_ptr);
                data_ptr += 1;
            }
            return data;
        }

        static const unsigned char DELETE_MARK = 0x01;
        // static const unsigned char REUSE_MARK = 0x10;
        /**
         * Marks an element with the given label deleted, does NOT really change the current graph.
         * @param label
         */
        void markDelete(labeltype label)
        {
            auto search = label_lookup_.find(label);
            if (search == label_lookup_.end()) {
                throw std::runtime_error("Label not found");
            }
            tableint internalId = search->second;
            markDeletedInternal(internalId);
        }

        /**
         * Uses the first 8 bits of the memory for the linked list to store the mark,
         * whereas maxM0_ has to be limited to the lower 24 bits, however, still large enough in almost all cases.
         * @param internalId
         */
        void markDeletedInternal(tableint internalId) {
            assert(internalId < cur_element_count);
            if (!isMarkedDeleted(internalId))
            {
                unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId))+2;
                *ll_cur |= DELETE_MARK;
                num_deleted_ += 1;
            }
            else
            {
                throw std::runtime_error("The requested to delete element is already deleted");
            }
        }

        /**
         * Remove the deleted mark of the node, does NOT really change the current graph.
         * @param label
         */
        void unmarkDelete(labeltype label)
        {
            auto search = label_lookup_.find(label);
            if (search == label_lookup_.end()) {
                throw std::runtime_error("Label not found");
            }
            tableint internalId = search->second;
            unmarkDeletedInternal(internalId);
        }

        /**
         * Remove the deleted mark of the node.
         * @param internalId
         */
        void unmarkDeletedInternal(tableint internalId) {
            assert(internalId < cur_element_count);
            if (isMarkedDeleted(internalId))
            {
                unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId))+2;
                *ll_cur &= ~DELETE_MARK;
                num_deleted_ -= 1;
            }
            else
            {
                throw std::runtime_error("The requested to undelete element is not deleted");
            }
        }

        /**
         * Checks the first 8 bits of the memory to see if the element is marked deleted.
         * @param internalId
         * @return
         */
        bool isMarkedDeleted(tableint internalId) const {//用于判断*ll_cur是否为0的方法，不改变*ll_cur的值
            unsigned char *ll_cur = ((unsigned char*)get_linklist0(internalId))+2;//是flag,删除标记(如果认为offsetLevel0_是0的话）
            return *ll_cur & DELETE_MARK;//DELETE_MARK=16,因为1046可认为*ll_cur初始值为0，按位与后返回0
        }

        unsigned short int getListCount(linklistsizeint * ptr) const {
            return *((unsigned short int *)ptr);
        }

        void setListCount(linklistsizeint * ptr, unsigned short int size) const {
            *((unsigned short int*)(ptr))=*((unsigned short int *)&size);// unsigned short int是两字节，也就是header部分的size
        }

        void addPoint(const void *data_point, labeltype label) {
            addPoint(data_point, label,-1);
        }
       
        /*
        更新节点， 这个与添加节点类似：
	    1. 从当前图的从最高层逐层往下寻找直至节点的层数+1停止，寻找到离data_point最近的节点，作为下面一层寻找的起始点。
	    2. 从data_point的最高层依次开始往下，每一层寻找离data_point最接近的ef_construction_（构建HNSW是可指定）个节点构成候选集，再从候选集中利用启发式搜索选择M个节点与data_point相互连接.
        */
        void updatePoint(const void *dataPoint, tableint internalId, float updateNeighborProbability) {
            // update the feature vector associated with existing point with new vector
            memcpy(getDataByInternalId(internalId), dataPoint, data_size_);

            int maxLevelCopy = maxlevel_;
            tableint entryPointCopy = enterpoint_node_;
            // If point to be updated is entry point and graph just contains single element then just return.
            if (entryPointCopy == internalId && cur_element_count == 1)
                return;

            int elemLevel = element_levels_[internalId];
            std::uniform_real_distribution<float> distribution(0.0, 1.0);


            //从第0层逐层往上，直至该节点的最高层，在每一层取待更新节点的部分邻居，更新他们的邻居。
            //for循环遍历每一层，在for循环里面，首先挑部分原来的邻居，存储在 sNeigh里面， 比例由参数updateNeighborProbability控制。
            //而将待更新节点经过一跳，二跳到达的节点存在sCand里面，供后面更新邻居的时候选择。
            for (int layer = 0; layer <= elemLevel; layer++) {
                std::unordered_set<tableint> sCand;
                std::unordered_set<tableint> sNeigh;
                std::vector<tableint> listOneHop = getConnectionsWithLock(internalId, layer);
                if (listOneHop.size() == 0)
                    continue;

                sCand.insert(internalId);

                for (auto&& elOneHop : listOneHop) {
                    sCand.insert(elOneHop);

                    if (distribution(update_probability_generator_) > updateNeighborProbability)
                        continue;

                    sNeigh.insert(elOneHop);

                    std::vector<tableint> listTwoHop = getConnectionsWithLock(elOneHop, layer);
                    for (auto&& elTwoHop : listTwoHop) {
                        sCand.insert(elTwoHop);
                    }
                }
                //然后，对sNeigh中每一个选中的待更新的邻居n，利用启发式搜索(getNeighborsByHeuristic2)在sCand中选出最多M个点，将它们作为n的邻居存储在n的数据结构对应的位置。
                for (auto&& neigh : sNeigh) {
                    // if (neigh == internalId)
                    //     continue;

                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                    size_t size = sCand.find(neigh) == sCand.end() ? sCand.size() : sCand.size() - 1; // sCand guaranteed to have size >= 1
                    size_t elementsToKeep = std::min(ef_construction_, size);
                    for (auto&& cand : sCand) {
                        if (cand == neigh)
                            continue;

                        dist_t distance = fstdistfunc_(getDataByInternalId(neigh), getDataByInternalId(cand), dist_func_param_);
                        if (candidates.size() < elementsToKeep) {
                            candidates.emplace(distance, cand);
                        } else {
                            if (distance < candidates.top().first) {
                                candidates.pop();
                                candidates.emplace(distance, cand);
                            }
                        }
                    }

                    // Retrieve neighbours using heuristic and set connections.
                    // 对于sNeigh中的每一个选中的待更新的邻居n,利用启发式搜索在sCand中选出最多M个点
                    getNeighborsByHeuristic2(candidates, layer == 0 ? maxM0_ : maxM_);

                    {
                        std::unique_lock <std::mutex> lock(link_list_locks_[neigh]);
                        linklistsizeint *ll_cur;
                        ll_cur = get_linklist_at_level(neigh, layer);
                        size_t candSize = candidates.size();
                        setListCount(ll_cur, candSize);
                        tableint *data = (tableint *) (ll_cur + 1);
                        for (size_t idx = 0; idx < candSize; idx++) {
                            data[idx] = candidates.top().second;
                            candidates.pop();
                        }
                    }
                }
            }
           //第三步，更新待更新节点data_point的邻居。这个与添加节点类似：从当前图的从最高层逐层往下寻找直至节点的层数+1停止，寻找到离data_point最近的节点，作为下面一层寻找的起始点。
           //2）从data_point的最高层依次开始往下，每一层寻找离data_point最接近的ef_construction_（构建HNSW是可指定）个节点构成候选集，再从候选集中利用启发式搜索选择M个节点与data_point相互连接.
            repairConnectionsForUpdate(dataPoint, entryPointCopy, internalId, elemLevel, maxLevelCopy);
        };

        //更新待更新节点dataPoint的邻居
        void repairConnectionsForUpdate(const void *dataPoint, tableint entryPointInternalId, tableint dataPointInternalId, int dataPointLevel, int maxLevel) {
            tableint currObj = entryPointInternalId;
            if (dataPointLevel < maxLevel) {
                dist_t curdist = fstdistfunc_(dataPoint, getDataByInternalId(currObj), dist_func_param_);
                //从最高层往下寻找直至节点的层数+1，寻找到离dataPoint最近的节点，作为下面一层寻找的起始点
                for (int level = maxLevel; level > dataPointLevel; level--) {
                    bool changed = true;
                    while (changed) {
                        changed = false;
                        unsigned int *data;
                        std::unique_lock <std::mutex> lock(link_list_locks_[currObj]);
                        data = get_linklist_at_level(currObj,level);
                        int size = getListCount(data);
                        tableint *datal = (tableint *) (data + 1);//指向了data第一个邻居的id
#ifdef USE_SSE
                        _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
#endif
                        for (int i = 0; i < size; i++) {
#ifdef USE_SSE
                            _mm_prefetch(getDataByInternalId(*(datal + i + 1)), _MM_HINT_T0);
#endif
                            tableint cand = datal[i];
                            dist_t d = fstdistfunc_(dataPoint, getDataByInternalId(cand), dist_func_param_);
                            if (d < curdist) {
                                curdist = d;
                                currObj = cand;
                                changed = true;
                            }
                        }
                    }
                }
            }

            if (dataPointLevel > maxLevel)
                throw std::runtime_error("Level of item to be updated cannot be bigger than max level");
            //从dataPoint层到最底层，搜索出topCandidates，然后除去自己，存到filteredTopCandidates里面
            for (int level = dataPointLevel; level >= 0; level--) {
                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> topCandidates = searchBaseLayer(
                        currObj, dataPoint, level);

                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> filteredTopCandidates;
                while (topCandidates.size() > 0) {
                    if (topCandidates.top().second != dataPointInternalId)
                        filteredTopCandidates.push(topCandidates.top());

                    topCandidates.pop();
                }

                // Since element_levels_ is being used to get `dataPointLevel`, there could be cases where `topCandidates` could just contains entry point itself.
                // To prevent self loops, the `topCandidates` is filtered and thus can be empty.
                if (filteredTopCandidates.size() > 0) {
                    bool epDeleted = isMarkedDeleted(entryPointInternalId);
                    if (epDeleted) {
                        filteredTopCandidates.emplace(fstdistfunc_(dataPoint, getDataByInternalId(entryPointInternalId), dist_func_param_), entryPointInternalId);
                        if (filteredTopCandidates.size() > ef_construction_)
                            filteredTopCandidates.pop();
                    }

                    currObj = mutuallyConnectNewElement(dataPoint, dataPointInternalId, filteredTopCandidates, level, true);//mutuallyConnectNewElement里面包含启发式搜索，并连接
                }
            }
        }

        std::vector<tableint> getConnectionsWithLock(tableint internalId, int level) {//获取指定level的邻居，存到一个vector里面
            std::unique_lock <std::mutex> lock(link_list_locks_[internalId]);
            unsigned int *data = get_linklist_at_level(internalId, level);
            int size = getListCount(data);
            std::vector<tableint> result(size);
            tableint *ll = (tableint *) (data + 1);
            memcpy(result.data(), ll,size * sizeof(tableint));
            return result;
        };

        tableint addPoint(const void *data_point, labeltype label, int level) {

            tableint cur_c = 0;
            {   //添加节点时，先检查一下该节点的label是否已经存在了，如果存在的话，直接更新节点
                // Checking if the element with the same label already exists
                // if so, updating it *instead* of creating a new element.
                std::unique_lock <std::mutex> templock_curr(cur_element_count_guard_);
                auto search = label_lookup_.find(label);
                if (search != label_lookup_.end()) {
                    tableint existingInternalId = search->second;
                    templock_curr.unlock();

                    std::unique_lock <std::mutex> lock_el_update(link_list_update_locks_[(existingInternalId & (max_update_element_locks - 1))]);//锁，控制节点更新

                    if (isMarkedDeleted(existingInternalId)) {
                        unmarkDeletedInternal(existingInternalId);
                    }
                    updatePoint(data_point, existingInternalId, 1.0);//更新节点
                    
                    return existingInternalId;
                }

                if (cur_element_count >= max_elements_) {
                    throw std::runtime_error("The number of elements exceeds the specified limit");
                };
               //如果图中不存在这个节点，首先确定该节点的index也就是cur_c,赋值为元素个数，这个是连续的
                cur_c = cur_element_count;
                cur_element_count++;  //节点id自增加1
                label_lookup_[label] = cur_c;
            }

            // Take update lock to prevent race conditions on an element with insertion/update at the same time.
            std::unique_lock <std::mutex> lock_el_update(link_list_update_locks_[(cur_c & (max_update_element_locks - 1))]);
            std::unique_lock <std::mutex> lock_el(link_list_locks_[cur_c]);
            int curlevel = getRandomLevel(mult_); //随机初始化层数 curlevel, 确定插入点所在的层数，从这层开始，到底层，这个点都需要出现。
            if (level > 0)
                curlevel = level;

            element_levels_[cur_c] = curlevel;//存储每个元素的level,数组


            std::unique_lock <std::mutex> templock(global);
            int maxlevelcopy = maxlevel_;
            if (curlevel <= maxlevelcopy)
                templock.unlock();
            tableint currObj = enterpoint_node_;
            tableint enterpoint_copy = enterpoint_node_;

            //初始化节点相关数据结构，主要包括：将节点数据以及label拷贝到第0层数据结构(data_level0_memory_)中；为该节点分配存储0层以上的邻居关系的结构，并将其地址存储在linkLists_中
            memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_);

            // Initialisation of the data and label
            memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));//getExternalLabeLp是计算这个cur_c这个节点的label所在的内存， 然后给他初始化
            memcpy(getDataByInternalId(cur_c), data_point, data_size_);//getDataByInternalId是计算这个cur_c这个节点的data所在的内存，然后给他初始化

            //为该节点分配存储0层以上的邻居关系的结构，并将其地址存储在linkLists_中
            if (curlevel) {
                linkLists_[cur_c] = (char *) malloc(size_links_per_element_ * curlevel + 1);
                if (linkLists_[cur_c] == nullptr)
                    throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
                memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel + 1);
            }
            //待添加的节点不是第一个元素
            //1）那么从当前图的从最高层逐层往下寻找直至节点的层数+1停止，寻找到离data_point最近的节点，作为下面一层寻找的起始点。
            //2）从curlevel依次开始往下，每一层寻找离data_point最接近的ef_construction_（构建HNSW是可指定）个节点构成候选集，再从候选集中选择M个节点与data_point相互连接。
            if ((signed)currObj != -1) {

                if (curlevel < maxlevelcopy) {

                    dist_t curdist = fstdistfunc_(data_point, getDataByInternalId(currObj), dist_func_param_);
                    for (int level = maxlevelcopy; level > curlevel; level--) {
                        //1. 逐层往下寻找直至curlevel+1，找到最近的一个节点curObj作为q插入到l层的入口点

                        bool changed = true;
                        while (changed) {
                            changed = false;
                            unsigned int *data;
                            std::unique_lock <std::mutex> lock(link_list_locks_[currObj]);
                            data = get_linklist(currObj,level);
                            int size = getListCount(data);

                            tableint *datal = (tableint *) (data + 1);//指向了data第一个邻居的id
                            for (int i = 0; i < size; i++) {
                                tableint cand = datal[i];
                                if (cand < 0 || cand > max_elements_)
                                    throw std::runtime_error("cand error");
                                dist_t d = fstdistfunc_(data_point, getDataByInternalId(cand), dist_func_param_);
                                if (d < curdist) { //打擂台，找到所有层中距离data_point最近的一个点作为data_point插入level层的入口点。
                                    curdist = d;
                                    currObj = cand;
                                    changed = true;
                                }
                            }
                        }
                    }
                }

                bool epDeleted = isMarkedDeleted(enterpoint_copy);
                for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--) {
                    //2. 从curlevel往下，找一定数量的邻居并连接(每一层找最多ef_construction个点，然后连接)
                    if (level > maxlevelcopy || level < 0)  // possible?
                        throw std::runtime_error("Level error");

                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates = searchBaseLayer(
                            currObj, data_point, level);
                    if (epDeleted) {
                        top_candidates.emplace(fstdistfunc_(data_point, getDataByInternalId(enterpoint_copy), dist_func_param_), enterpoint_copy);
                        if (top_candidates.size() > ef_construction_) //大于ef_construction_
                            top_candidates.pop();
                    }
                    currObj = mutuallyConnectNewElement(data_point, cur_c, top_candidates, level, false);//mutuallyConnectNewElement里面包含选M——个元素以及启发式算法裁边
                }


            } else {  //如果这是第一个元素，只需将该节点作为HNSW的entrypoint，并将该元素的层数作为当前的最大层。（enterpoint_node_ =0; maxlevel_ = curlevel）
                // Do nothing for the first element
                enterpoint_node_ = 0;
                maxlevel_ = curlevel;

            }

            //Releasing lock for the maximum level
            if (curlevel > maxlevelcopy) {
                enterpoint_node_ = cur_c;
                maxlevel_ = curlevel;
            }
            return cur_c;
        };
//    #if 0
//         std::priority_queue<std::pair<dist_t, labeltype >>
//         searchKnn(const void *query_data, size_t k) const {
//             std::priority_queue<std::pair<dist_t, labeltype >> result;
//             if (cur_element_count == 0) return result;
//             //currobj和curdist分别记录距离data point最近的点和距离
//             tableint currObj = enterpoint_node_;
//             dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);
//             //在层L...1之间
//             for (int level = maxlevel_; level > 0; level--) {
//                 bool changed = true;
//                 while (changed) {
//                     //首先没有变化，表示在同一层中搜索
//                     changed = false;
//                     unsigned int *data;
//                     //获得currObj的连接数，也就是邻居
//                     data = (unsigned int *) get_linklist(currObj, level);
//                     int size = getListCount(data);
//                     metric_hops++;
//                     metric_distance_computations+=size;

//                     tableint *datal = (tableint *) (data + 1);//指向了data第一个邻居的id, datal表示当前元素第一个邻居的label
//                     //对于currObj的每一个邻居，计算它与query_data的距离，并及时更新currObj和currdist
//                     for (int i = 0; i < size; i++) {
//                         //获取邻居id
//                         tableint cand = datal[i];
//                         if (cand < 0 || cand > max_elements_)
//                             throw std::runtime_error("cand error");
//                         //根据id获取邻居并计算其到query的距离
//                         dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);
//                         //如果这个邻居与query的距离比curdist还要小，更新curdist为这个邻居，changed改为true
//                         if (d < curdist) {
//                             curdist = d;
//                             currObj = cand;
//                             changed = true;
//                         }
//                     }
//                 }
//             }
//             //目前已经获得第一层与query最近的元素currObj
//             //在第0层获取currObj邻居中距离query最近的max(ef_, k)个邻居，也就是动态列表top_candidates，论文中是W
//             std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
//             if (num_deleted_) {
//                 top_candidates=searchBaseLayerST<true,true>(// has_deletions:true, collect_metrics: true
//                         currObj, query_data, std::max(ef_, k));
//             }
//             else{
//                 top_candidates=searchBaseLayerST<false,true>(// has_deletions:false, collect_metrics: true
//                         currObj, query_data, std::max(ef_, k));
//             }
//             //top_candidates修剪为k个
//             while (top_candidates.size() > k) {
//                 top_candidates.pop();
//             }
//             //结果存到result中
//             while (top_candidates.size() > 0) {
//                 std::pair<dist_t, tableint> rez = top_candidates.top();
//                 result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
//                 top_candidates.pop();
//             }
//             return result;
//         };

//     #endif
        //multi-search
        // #if 1
        std::priority_queue<std::pair<dist_t, labeltype >>
        searchKnn(const void *query_data, size_t k) const {
            std::priority_queue<std::pair<dist_t, labeltype >> result;
            if (cur_element_count == 0) return result;
            //currobj和curdist分别记录距离data point最近的点和距离
            tableint currObj = enterpoint_node_;
            dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);
            //在层L...1之间
            for (int level = maxlevel_; level > 1; level--) {
                bool changed = true;
                while (changed) {
                    //首先没有变化，表示在同一层中搜索
                    changed = false;
                    unsigned int *data;
                    //获得currObj的连接数，也就是邻居
                    data = (unsigned int *) get_linklist(currObj, level);
                    int size = getListCount(data);
                    metric_hops++;
                    metric_distance_computations+=size;

                    tableint *datal = (tableint *) (data + 1);//指向了data第一个邻居的id, datal表示当前元素第一个邻居的label
                    //对于currObj的每一个邻居，计算它与query_data的距离，并及时更新currObj和currdist
                    for (int i = 0; i < size; i++) {
                        //获取邻居id
                        tableint cand = datal[i];
                        if (cand < 0 || cand > max_elements_)
                            throw std::runtime_error("cand error");
                        //根据id获取邻居并计算其到query的距离
                        dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);
                        //如果这个邻居与query的距离比curdist还要小，更新curdist为这个邻居，changed改为true
                        if (d < curdist) {
                            curdist = d;
                            currObj = cand;
                            changed = true;
                        }
                    }
                }
            }
            // how to expand parallel search
            //method 1: m thread=cores in the beginning, so you need to find m enter_point in layer 1 for base layer
            // below is how to find m enter_point in layer 1, can not use searchBaseLayerST,for it only search in baseLayer, can not use searhBaseLayer, it will serach ef_construction points.
            int thread_num = 8;
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates1;
            top_candidates1 = searchLayerL1(currObj,query_data,thread_num);
            // std::cout<<"---------------------top_candidates1.size"<<top_candidates1.size()<<std::endl;
            std::vector<std::pair<dist_t,tableint>> seeds;
            // while(!top_candidates1.empty()){
            //     seeds.emplace_back(top_candidates1.top());
            //     top_candidates1.pop();
            // }

            // std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
            // std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;

            // std::vector<std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>> top_candidates_vector;
            // std::vector<std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>> candidate_set_vector;

           std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates_vector[thread_num];
           std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set_vector[thread_num];

            // while(!top_candidates1.empty()){
            //     tableint ep_id = top_candidates1.top().second;
            //     top_candidates1.pop();
            //     // std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates2 = searchBaseLayerST<false>(ep_id, query_data, ef_);
            //     std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates2 = parallelsearchBaseLayerST<false>(ep_id, query_data, ef_,thread_num);
            //     while(!top_candidates2.empty()){
            //         top_candidates.push(top_candidates2.top());
            //         top_candidates2.pop();
            //     }
            // }
            // int i;
            // #pragma omp parallel for private(i)
            omp_set_num_threads(thread_num);
            #pragma omp parallel for
            // for(i=0;i<seeds.size();i++){
            for(int i=0;i<top_candidates1.size();i++){
                // tableint pix =seeds[i].second;
                tableint pix = top_candidates1.top().second;
                // std::cout<<"pix--------------"<<pix<<std::endl;
                top_candidates1.pop();
                VisitedList *vl = visited_list_pool_->getFreeVisitedList();// VisitedList存储已访问过的节点，下面进行其初始化过程
                vl_type *visited_array = vl->mass; // 新建vl_type实例
                vl_type visited_array_tag = vl->curV; // visited_array_tag初始化为-1
                // std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
                // std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;
                dist_t lowerBound;
               //    num_deleted_统计flag=1的元素数，删除和 被访问不同
                if (!isMarkedDeleted(pix)) {//currObj没被删时或num_deleted_为0时进入
                    dist_t dist = fstdistfunc_(query_data, getDataByInternalId(pix), dist_func_param_);//q和data的距离
                    lowerBound = dist;
                    // std::cout<<"dist:"<<dist<<"pix: "<<pix<<std::endl;
                    top_candidates_vector[i].emplace(dist, pix);
                    candidate_set_vector[i].emplace(-dist, pix);
                    // top_candidates.emplace(dist, pix);
                    // candidate_set.emplace(-dist, pix);
                } else {
                    lowerBound = std::numeric_limits<dist_t>::max();
                    candidate_set_vector[i].emplace(-lowerBound, pix);//倒序排列
                    // candidate_set.emplace(-lowerBound, pix);
                }
                visited_array[pix] = visited_array_tag;
                
                // //这个时候candidate_set和top_candidates只有一个值

                // // visited_array[pix] = visited_array_tag;//存为1，表示已被访问   
                // if(visited_array[pix]==visited_array_tag || pix==-1)
                //     continue;
                // visited_array[pix] = visited_array_tag;
                // dist_t dist = fstdistfunc_(data_point, getDataByInternalId(pix), dist_func_param_);
                int step=0;
                while (!candidate_set_vector[i].empty()) {
                // while(!candidate_set.empty()){

                size_t ef_staged =0;
                ef_staged == ef_ ?ef_: step++;

                std::pair<dist_t, tableint> current_node_pair = candidate_set_vector[i].top();
                // std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
                // if ((-current_node_pair.first) > lowerBound && (top_candidates.size() == ef_)) {
                if ((-current_node_pair.first) > lowerBound && (top_candidates_vector[i].size() == ef_)) { //每个local_thread搜索的长度现在设置的也是ef_
                    break;
                }
                candidate_set_vector[i].pop();
                // candidate_set.pop();

                tableint current_node_id = current_node_pair.second;
                int *data = (int *) get_linklist0(current_node_id);
                size_t size = getListCount((linklistsizeint*)data);
                // std::cout<<"size:"<<size<<std::endl;
//                bool cur_node_deleted = isMarkedDeleted(current_node_id);


#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
                _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
                _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
                _mm_prefetch((char *) (data + 2), _MM_HINT_T0);
#endif

                for (size_t j = 1; j <= size; j++) {
                    int candidate_id = *(data + j);
                    // std::cout<<"candidate_id:"<<candidate_id<<std::endl;
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                    _mm_prefetch((char *) (visited_array + *(data + j + 1)), _MM_HINT_T0);
                    _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_,
                                 _MM_HINT_T0);////////////
#endif
                    if (!(visited_array[candidate_id] == visited_array_tag)) {

                        visited_array[candidate_id] = visited_array_tag;

                        char *currObj1 = (getDataByInternalId(candidate_id));
                        dist_t dist = fstdistfunc_(query_data, currObj1, dist_func_param_);
                        // std::cout<<"dist:::::"<<dist<<std::endl;

                        if (top_candidates_vector[i].size() < ef_ || lowerBound > dist) {
                            candidate_set_vector[i].emplace(-dist, candidate_id);
                        // if (top_candidates.size() < ef_ || lowerBound > dist) {
                        //     candidate_set.emplace(-dist, candidate_id);
#ifdef USE_SSE
                            _mm_prefetch(data_level0_memory_ + candidate_set_vector[i].top().second * size_data_per_element_ +
                                         offsetLevel0_,///////////
                                         _MM_HINT_T0);////////////////////////
                            // _mm_prefetch(data_level0_memory_ + candidate_set.top().second * size_data_per_element_ +
                            //                 offsetLevel0_,///////////
                            //                 _MM_HINT_T0);////////////////////////
#endif

                            if ( !isMarkedDeleted(candidate_id))
                                top_candidates_vector[i].emplace(dist, candidate_id);

                            if (top_candidates_vector[i].size() > ef_)
                                top_candidates_vector[i].pop();

                            if (!top_candidates_vector[i].empty())
                                lowerBound = top_candidates_vector[i].top().first;


                            // if ( !isMarkedDeleted(candidate_id))
                            //     top_candidates.emplace(dist, candidate_id);

                            // if (top_candidates.size() > ef_)
                            //     top_candidates.pop();

                            // if (!top_candidates.empty())
                            //     lowerBound = top_candidates.top().first;

                            // std::cout<<"end----"<<std::endl;
                        }
                    }
                }
            }

            visited_list_pool_->releaseVisitedList(vl);


        // top_candidates_vector.emplace_back(top_candidates);
        // candidate_set_vector.emplace_back(candidate_set);

        // top_candidates_vector.emplace_back(top_candidates_vector[i]);
        // candidate_set_vector.emplace_back(candidate_set_vector[i]);
       

        }
        

#pragma omp barrier
        //接下来这边需要的是合并各个thread的结果,合并结果并截取k个

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        // std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;

       /*  //不好使
        std::unordered_set<std::pair<dist_t, tableint>> top_candidates_set;//利用set去重
        //合并thread的结果，存到top_candidates里面
        
        for(int i=0;i<top_candidates_vector.size();i++){
            // candidate_set.push_back(top_candidates_vector[i]);
            while(!top_candidates_vector[i].empty()){
                top_candidates_set.insert(top_candidates_vector[i].top());
                top_candidates_vector[i].pop();
            }
        }

        for(auto i:top_candidates_set){
            top_candidates.emplace_back(i);
        }

        */
    //    std::vector<tableint> temp_set;
       tableint temp_set[1000000];
       memset(temp_set, 0, sizeof(tableint));
    //    for(int i=0;i<top_candidates_vector.size();i++){
        for(int i=0;i<thread_num;i++){
        // std::cout<<"top_candidates_vector[i].size()"<<top_candidates_vector[i].size()<<std::endl;
        while(top_candidates_vector[i].size()){
            // for(int j=0;j<top_candidates_vector[i].size();j++){
                tableint temp = top_candidates_vector[i].top().second;
            // std::cout<<"temp"<<temp<<std::endl;
            // if(temp_set.find(temp)==temp_set.end()){
            //     temp_set.insert({temp,1});
            // }
            // else{
            //     temp_set[temp]

            // }
            // temp_set.emplace_back(temp);
            temp_set[temp]++;
            // std::cout<<"temp_set[temp]: "<<temp_set[temp]<<std::endl;
            // printf("temp_set[temp]:%d.\n", temp_set[temp]);
            // if(temp_set.find(temp)!=temp_set.end()){
            //这块的去重的不太合理呀，但是结果是好的
            if(temp_set[temp]>1){
                top_candidates.emplace(top_candidates_vector[i].top());
                top_candidates_vector[i].pop();
            }
            else if(temp_set[temp]==1){
                top_candidates_vector[i].pop();
            }

        }
        // std::cout<<"debug  top_candidates size:"<<top_candidates.size()<<std::endl;
        // while(!top_candidates_vector[i].empty()){
        //        if(temp_set[temp]!=1){
        //         top_candidates.emplace(top_candidates_vector[i].top());
        //         // std::cout<<"top_candidates_vector[i]"<<top_candidates_vector[i].top().second<<std::endl;
        //         top_candidates_vector[i].pop();
        //     }
        // }
       }
    //    std::cout<<"debug"<<std::endl;
        //top_candidates修剪为k个
        while (top_candidates.size() > k) {
            top_candidates.pop();
        }
        //结果存到result中
        while (top_candidates.size() > 0) {
            std::pair<dist_t, tableint> rez = top_candidates.top();
            result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
            top_candidates.pop();
        }
        return result;
        };

        // #endif

        void checkIntegrity(){
            int connections_checked=0;
            std::vector <int > inbound_connections_num(cur_element_count,0);
            for(int i = 0;i < cur_element_count; i++){
                for(int l = 0;l <= element_levels_[i]; l++){
                    linklistsizeint *ll_cur = get_linklist_at_level(i,l);
                    int size = getListCount(ll_cur);
                    tableint *data = (tableint *) (ll_cur + 1);
                    std::unordered_set<tableint> s;
                    for (int j=0; j<size; j++){
                        assert(data[j] > 0);
                        assert(data[j] < cur_element_count);
                        assert (data[j] != i);
                        inbound_connections_num[data[j]]++; //inbound_connections_num存储的是元素数对应的连接数
                        s.insert(data[j]);
                        connections_checked++;//连接边数

                    }
                    assert(s.size() == size);
                }
            }
            if(cur_element_count > 1){
                int min1=inbound_connections_num[0], max1=inbound_connections_num[0]; //min1, max1存储的是元素的最小最大值
                for(int i=0; i < cur_element_count; i++){
                    assert(inbound_connections_num[i] > 0);
                    min1=std::min(inbound_connections_num[i],min1);
                    max1=std::max(inbound_connections_num[i],max1);
                }
                std::cout << "Min inbound: " << min1 << ", Max inbound:" << max1 << "\n";
            }
            std::cout << "integrity ok, checked " << connections_checked << " connections\n";

        }

    };

}
