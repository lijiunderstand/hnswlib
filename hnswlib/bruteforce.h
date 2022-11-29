#pragma once
#include <unordered_map>
#include <fstream>
#include <mutex>
#include <algorithm>

namespace hnswlib {
    template<typename dist_t>
    class BruteforceSearch : public AlgorithmInterface<dist_t> {
    public:
        BruteforceSearch(SpaceInterface <dist_t> *s) {

        }
        BruteforceSearch(SpaceInterface<dist_t> *s, const std::string &location) {
            loadIndex(location, s);
        }

        BruteforceSearch(SpaceInterface <dist_t> *s, size_t maxElements) {
            maxelements_ = maxElements;
            data_size_ = s->get_data_size();
            fstdistfunc_ = s->get_dist_func();
            dist_func_param_ = s->get_dist_func_param();
            size_per_element_ = data_size_ + sizeof(labeltype);
            data_ = (char *) malloc(maxElements * size_per_element_);
            if (data_ == nullptr)
                std::runtime_error("Not enough memory: BruteforceSearch failed to allocate data");
            cur_element_count = 0;
        }

        ~BruteforceSearch() {
            free(data_);
        }

        char *data_;
        size_t maxelements_;
        size_t cur_element_count;
        size_t size_per_element_;

        size_t data_size_; //float或者是int之类的
        DISTFUNC <dist_t> fstdistfunc_;
        void *dist_func_param_;
        std::mutex index_lock;

        std::unordered_map<labeltype,size_t > dict_external_to_internal;

        void addPoint(const void *datapoint, labeltype label) {

            int idx;
            {
                std::unique_lock<std::mutex> lock(index_lock);



                auto search=dict_external_to_internal.find(label); //label是外部的index
                if (search != dict_external_to_internal.end()) {
                    idx=search->second; //找到了，说明这个datapoint已经在图里面了
                }
                else{
                    if (cur_element_count >= maxelements_) {
                        throw std::runtime_error("The number of elements exceeds the specified limit\n");
                    }
                    idx=cur_element_count;//内部的index就是idx把他设置为count数（内部的点数）
                    dict_external_to_internal[label] = idx; //这个查找字典，键是外部的label，值是内部的index,也就是说把内外部index做一个映射
                    cur_element_count++;//已经建好的图里面的元素数量.
                }
            }
            memcpy(data_ + size_per_element_ * idx + data_size_, &label, sizeof(labeltype));//完成label的内存搬运
            memcpy(data_ + size_per_element_ * idx, datapoint, data_size_); //完成data的内存搬运.




        };

        void removePoint(labeltype cur_external) {
            size_t cur_c=dict_external_to_internal[cur_external]; //cur_external是外部的index, cur_c是内部的index

            dict_external_to_internal.erase(cur_external); //删除内部字典里面的这个键值对.

            labeltype label=*((labeltype*)(data_ + size_per_element_ * (cur_element_count-1) + data_size_));
            dict_external_to_internal[label]=cur_c;
            memcpy(data_ + size_per_element_ * cur_c,
                   data_ + size_per_element_ * (cur_element_count-1),
                   data_size_+sizeof(labeltype));
            cur_element_count--;

        }


        std::priority_queue<std::pair<dist_t, labeltype >>
        searchKnn(const void *query_data, size_t k) const {
            std::priority_queue<std::pair<dist_t, labeltype >> topResults;
            if (cur_element_count == 0) return topResults;
            for (int i = 0; i < k; i++) { //这边是先给topResults里面写k个数
                dist_t dist = fstdistfunc_(query_data, data_ + size_per_element_ * i, dist_func_param_);
                topResults.push(std::pair<dist_t, labeltype>(dist, *((labeltype *) (data_ + size_per_element_ * i +
                                                                                    data_size_))));
            }
            dist_t lastdist = topResults.top().first; //最小的dist
            for (int i = k; i < cur_element_count; i++) {
                dist_t dist = fstdistfunc_(query_data, data_ + size_per_element_ * i, dist_func_param_);
                if (dist <= lastdist) { //如果有更近的，更新topResults里面的内容.
                    topResults.push(std::pair<dist_t, labeltype>(dist, *((labeltype *) (data_ + size_per_element_ * i +
                                                                                        data_size_))));
                    if (topResults.size() > k)
                        topResults.pop();
                    lastdist = topResults.top().first; //优先队列的首元素.最小的dist
                }

            }
            return topResults;
        };

        void saveIndex(const std::string &location) {
            std::ofstream output(location, std::ios::binary);
            std::streampos position;

            writeBinaryPOD(output, maxelements_);
            writeBinaryPOD(output, size_per_element_);
            writeBinaryPOD(output, cur_element_count);

            output.write(data_, maxelements_ * size_per_element_);

            output.close();
        }

        void loadIndex(const std::string &location, SpaceInterface<dist_t> *s) {


            std::ifstream input(location, std::ios::binary);
            std::streampos position;

            readBinaryPOD(input, maxelements_);
            readBinaryPOD(input, size_per_element_);
            readBinaryPOD(input, cur_element_count);

            data_size_ = s->get_data_size();
            fstdistfunc_ = s->get_dist_func();
            dist_func_param_ = s->get_dist_func_param();
            size_per_element_ = data_size_ + sizeof(labeltype);
            data_ = (char *) malloc(maxelements_ * size_per_element_);
            if (data_ == nullptr)
                std::runtime_error("Not enough memory: loadIndex failed to allocate data");

            input.read(data_, maxelements_ * size_per_element_);

            input.close();

        }

    };
}
