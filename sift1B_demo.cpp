#include <iostream>
#include <fstream>
#include <queue>
#include <chrono>
#include "hnswlib/hnswlib.h"

#include <unordered_set>

using namespace std;
using namespace hnswlib;

class StopW
{
    std::chrono::steady_clock::time_point time_begin;

public:
    StopW()
    {
        time_begin = std::chrono::steady_clock::now();
    }

    float getElapsedTimeMicro()
    {
        std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
        return (std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin).count());
    }

    void reset()
    {
        time_begin = std::chrono::steady_clock::now();
    }
};

/*
 * Author:  David Robert Nadeau
 * Site:    http://NadeauSoftware.com/
 * License: Creative Commons Attribution 3.0 Unported License
 *          http://creativecommons.org/licenses/by/3.0/deed.en_US
 */

#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>

#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))

#include <unistd.h>
#include <sys/resource.h>

#if defined(__APPLE__) && defined(__MACH__)
#include <mach/mach.h>

#elif (defined(_AIX) || defined(__TOS__AIX__)) || (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
#include <fcntl.h>
#include <procfs.h>

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)

#endif

#else
#error "Cannot define getPeakRSS( ) or getCurrentRSS( ) for an unknown OS."
#endif

/**
 * Returns the peak (maximum so far) resident set size (physical
 * memory use) measured in bytes, or zero if the value cannot be
 * determined on this OS.
 */
static size_t getPeakRSS()
{
#if defined(_WIN32)
    /* Windows -------------------------------------------------- */
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
    return (size_t)info.PeakWorkingSetSize;

#elif (defined(_AIX) || defined(__TOS__AIX__)) || (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
    /* AIX and Solaris ------------------------------------------ */
    struct psinfo psinfo;
    int fd = -1;
    if ((fd = open("/proc/self/psinfo", O_RDONLY)) == -1)
        return (size_t)0L; /* Can't open? */
    if (read(fd, &psinfo, sizeof(psinfo)) != sizeof(psinfo))
    {
        close(fd);
        return (size_t)0L; /* Can't read? */
    }
    close(fd);
    return (size_t)(psinfo.pr_rssize * 1024L);

#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))
    /* BSD, Linux, and OSX -------------------------------------- */
    struct rusage rusage;
    getrusage(RUSAGE_SELF, &rusage);
#if defined(__APPLE__) && defined(__MACH__)
    return (size_t)rusage.ru_maxrss;
#else
    return (size_t)(rusage.ru_maxrss * 1024L);
#endif

#else
    /* Unknown OS ----------------------------------------------- */
    return (size_t)0L; /* Unsupported. */
#endif
}

/**
 * Returns the current resident set size (physical memory use) measured
 * in bytes, or zero if the value cannot be determined on this OS.
 */
static size_t getCurrentRSS()
{
#if defined(_WIN32)
    /* Windows -------------------------------------------------- */
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
    return (size_t)info.WorkingSetSize;

#elif defined(__APPLE__) && defined(__MACH__)
    /* OSX ------------------------------------------------------ */
    struct mach_task_basic_info info;
    mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  (task_info_t)&info, &infoCount) != KERN_SUCCESS)
        return (size_t)0L; /* Can't access? */
    return (size_t)info.resident_size;

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
    /* Linux ---------------------------------------------------- */
    long rss = 0L;
    FILE *fp = NULL;
    if ((fp = fopen("/proc/self/statm", "r")) == NULL)
        return (size_t)0L; /* Can't open? */
    if (fscanf(fp, "%*s%ld", &rss) != 1)
    {
        fclose(fp);
        return (size_t)0L; /* Can't read? */
    }
    fclose(fp);
    return (size_t)rss * (size_t)sysconf(_SC_PAGESIZE);

#else
    /* AIX, BSD, Solaris, and Unknown OS ------------------------ */
    return (size_t)0L; /* Unsupported. */
#endif
}

static void
get_gt(unsigned int *massQA, unsigned char *massQ, unsigned char *mass, size_t vecsize, size_t qsize, InnerProductSpaceI &ipspace,
       size_t vecdim, vector<std::priority_queue<std::pair<int, labeltype>>> &answers, size_t k)
{

    (vector<std::priority_queue<std::pair<int, labeltype>>>(qsize)).swap(answers);
    DISTFUNC<int> fstdistfunc_ = ipspace.get_dist_func();
    cout << qsize << "\n";
    for (int i = 0; i < qsize; i++)
    {
        for (int j = 0; j < k; j++)
        {
            answers[i].emplace(0.0f, massQA[1000 * i + j]);
        }
    }
}

static float
test_approx(unsigned char *massQ, size_t vecsize, size_t qsize, HierarchicalNSW<int> &appr_alg, size_t vecdim,
            vector<std::priority_queue<std::pair<int, labeltype>>> &answers, size_t k)
{
    size_t correct = 0;
    size_t total = 0;
    // uncomment to test in parallel mode:
    //#pragma omp parallel for
    for (int i = 0; i < qsize; i++)
    {

        std::priority_queue<std::pair<int, labeltype>> result = appr_alg.searchKnn(massQ + vecdim * i, k);
        std::priority_queue<std::pair<int, labeltype>> gt(answers[i]);
        unordered_set<labeltype> g;
        total += gt.size();

        while (gt.size())
        {

            g.insert(gt.top().second);
            gt.pop();
        }

        while (result.size())
        {
            if (g.find(result.top().second) != g.end())
            {

                correct++;
            }
            else
            {
            }
            result.pop();
        }
    }
    return 1.0f * correct / total;
}

static void
test_vs_recall(unsigned char *massQ, size_t vecsize, size_t qsize, HierarchicalNSW<int> &appr_alg, size_t vecdim,
               vector<std::priority_queue<std::pair<int, labeltype>>> &answers, size_t k)
{
    vector<size_t> efs; // = { 10,10,10,10,10 };
    for (int i = k; i < 30; i++)
    {
        efs.push_back(i);
    }
    for (int i = 30; i < 100; i += 10)
    {
        efs.push_back(i);
    }
    for (int i = 100; i < 1000; i += 40)
    {
        efs.push_back(i);
    }
    for (size_t ef : efs)
    {
        appr_alg.setEf(ef);
        StopW stopw = StopW();

        float recall = test_approx(massQ, vecsize, qsize, appr_alg, vecdim, answers, k);
        float time_us_per_query = stopw.getElapsedTimeMicro() / qsize;

        cout << ef << "\t" << recall << "\t" << time_us_per_query << " us\n";
        if (recall > 1.0)
        {
            cout << recall << "\t" << time_us_per_query << " us\n";
            break;
        }
    }
}

inline bool exists_test(const std::string &name)
{
    ifstream f(name.c_str());
    return f.good();
}

using stdclock = std::chrono::high_resolution_clock;
void sift_1b_demo()
{

    int subset_size_milllions = 200;   //subset for calculation, in this case, we use 2,0000,0000. for the whole dataset, subset_size is 1000
    int efConstruction = 40;
    int M = 16;

    size_t vecsize = subset_size_milllions * 1000000;

    // size_t qsize = 32;
    size_t qsize =10000;
    size_t vecdim = 128;
    char path_index[1024];
    char path_gt[1024];
    char *path_q = "/data/bigann/sift1b/queries.bvecs";
    char *path_data = "/data/bigann/sift1b/bigann_base.bvecs"; //xb
    sprintf(path_index, "sift1b_%dm_ef_%d_M_%d_IP.bin", subset_size_milllions, efConstruction, M); // buildIndex, path and filename to save index

    // sprintf(path_gt, "/home/intel/workspace/lijie/data/bigann/sift1b/gnd/idx_%dM.ivecs", subset_size_milllions);  //gt
    sprintf(path_gt, "/data/bigann/sift1b/gnd/idx_%dM.ivecs", subset_size_milllions);  //gt

    unsigned char *massb = new unsigned char[vecdim];  

    cout << "Loading GT:\n";
    ifstream inputGT(path_gt, ios::binary);
    unsigned int *massQA = new unsigned int[qsize * 1000];//sift1b gt: top1000
    for (int i = 0; i < qsize; i++)
    {
        int t;
        inputGT.read((char *)&t, 4);
        inputGT.read((char *)(massQA + 1000 * i), t * 4);
        if (t != 1000)
        {
            cout << "err";
            return;
        }
    }
    inputGT.close();

    cout << "Loading queries:\n";
    unsigned char *massQ = new unsigned char[qsize * vecdim];
    ifstream inputQ(path_q, ios::binary);

    for (int i = 0; i < qsize; i++)
    {
        int in = 0;
        inputQ.read((char *)&in, 4);
        if (in != 128)
        {
            cout << "file error";
            exit(1);
        }
        inputQ.read((char *)massb, in);
        for (int j = 0; j < vecdim; j++)
        {
            massQ[i * vecdim + j] = massb[j];
        }
    }
    inputQ.close();

    unsigned char *mass = new unsigned char[vecdim];
    ifstream input(path_data, ios::binary);
    int in = 0;
    InnerProductSpaceI ipspace(vecdim);

    HierarchicalNSW<int> *appr_alg;
    if (exists_test(path_index))
    {
        cout << "Loading index from " << path_index << ":\n";
        appr_alg = new HierarchicalNSW<int>(&ipspace, path_index, false); // loadIndex
        cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb \n";
    }
    else
    {
        cout << "Building index:\n";
        stdclock::time_point before = stdclock::now();
        appr_alg = new HierarchicalNSW<int>(&ipspace, vecsize, M, efConstruction);

        input.read((char *)&in, 4);
        if (in != 128)
        {
            cout << "file error";
            exit(1);
        }
        input.read((char *)massb, in);

        for (int j = 0; j < vecdim; j++)
        {
            mass[j] = massb[j] * (1.0f);
        }

        appr_alg->addPoint((void *)(massb), (size_t)0);
        int j1 = 0;
        StopW stopw = StopW();
        StopW stopw_full = StopW();
        size_t report_every = 100000;
#pragma omp parallel for
        for (int i = 1; i < vecsize; i++)
        {
            unsigned char mass[128];
            int j2 = 0;
#pragma omp critical
            {

                input.read((char *)&in, 4);
                if (in != 128)
                {
                    cout << "file error";
                    exit(1);
                }
                input.read((char *)massb, in);
                for (int j = 0; j < vecdim; j++)
                {
                    mass[j] = massb[j];
                }
                j1++;
                j2 = j1;
                if (j1 % report_every == 0)
                {
                    cout << j1 / (0.01 * vecsize) << " %, "
                         << report_every / (1000.0 * 1e-6 * stopw.getElapsedTimeMicro()) << " kips "
                         << " Mem: "
                         << getCurrentRSS() / 1000000 << " Mb \n";
                    stopw.reset();
                }
            }
            appr_alg->addPoint((void *)(mass), (size_t)j2);  //mass :single vector
        }
        input.close();
        cout << "Build time:" << 1e-6 * stopw_full.getElapsedTimeMicro() << "  seconds\n";
        appr_alg->saveIndex(path_index);

        stdclock::time_point after = stdclock::now();
        double search_cost = (std::chrono::duration<double>(after - before)).count();
        std::cout << "build index time spend: " << search_cost << std::endl;
    }

    vector<std::priority_queue<std::pair<int, labeltype>>> answers;
    size_t k = 1;
    std::cout<<"k:"<<k<<std::endl;
    // cout << "Parsing gt:\n";
    // get_gt(massQA, massQ, mass, vecsize, qsize, ipspace, vecdim, answers, k);  //get gt from file
    // cout << "Loaded gt\n";
    // for (int i = 0; i < 1; i++)
    //     test_vs_recall(massQ, vecsize, qsize, *appr_alg, vecdim, answers, k);


    for(int i=0;i<qsize;i++){
        // std::priority_queue<std::pair<float, labeltype>> result = appr_alg->searchKnn(massQ + vecdim * i, k);
        answers.push_back(appr_alg->searchKnn(massQ + vecdim * i, k));
    }
    std::cout<<"result size"<< answers.size()<<std::endl;

    cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb \n";
    return;
}


int main()
{
    sift_1b_demo();
    return 1;
}
