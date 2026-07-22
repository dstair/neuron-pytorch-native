#define _GNU_SOURCE

#include <dlfcn.h>
#include <execinfo.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <nrt/nrt.h>

typedef NRT_STATUS (*nrt_get_instance_info_fn)(
    nrt_instance_info_t *info,
    size_t instance_info_len);

static pthread_once_t resolve_once = PTHREAD_ONCE_INIT;
static nrt_get_instance_info_fn real_nrt_get_instance_info;
static void *nrt_handle;

static void resolve_nrt_get_instance_info(void) {
    void *symbol = dlsym(RTLD_NEXT, "nrt_get_instance_info");

    if (symbol == NULL) {
        nrt_handle = dlopen("libnrt.so.1", RTLD_NOW | RTLD_LOCAL);
        if (nrt_handle != NULL) {
            symbol = dlsym(nrt_handle, "nrt_get_instance_info");
        }
    }
    memcpy(&real_nrt_get_instance_info, &symbol, sizeof(symbol));
}

static int called_from_cache_key_constructor(void) {
    void *frames[16];
    int frame_count = backtrace(frames, (int)(sizeof(frames) / sizeof(frames[0])));
    int saw_cache_key_constructor = 0;
    int saw_compile_only_constructor = 0;
    const char *trace_stack = getenv("QWEN35_PLATFORM_TARGET_SHIM_TRACE_STACK");

    for (int index = 1; index < frame_count; ++index) {
        Dl_info frame = {0};

        if (dladdr(frames[index], &frame) != 0) {
            if (trace_stack != NULL && strcmp(trace_stack, "0") != 0) {
                fprintf(
                    stderr,
                    "qwen35-platform-frame index=%d object=%s symbol=%s\n",
                    index,
                    frame.dli_fname != NULL ? frame.dli_fname : "unknown",
                    frame.dli_sname != NULL ? frame.dli_sname : "unknown");
            }
            if (frame.dli_fname == NULL || frame.dli_sname == NULL ||
                strstr(frame.dli_fname, "torch_neuronx/_C") == NULL) {
                continue;
            }
            if (strstr(
                    frame.dli_sname,
                    "_ZN2at6neuron19CompilationCacheKeyC") != NULL) {
                saw_cache_key_constructor = 1;
            }
            if (strstr(
                    frame.dli_sname,
                    "_ZN2at6neuron26CompileOnlyKernelExecutionC") != NULL) {
                saw_compile_only_constructor = 1;
            }
        }
    }
    if (trace_stack != NULL && strcmp(trace_stack, "0") != 0) {
        fprintf(
            stderr,
            "qwen35-platform-stack cache_key=%d compile_only=%d\n",
            saw_cache_key_constructor,
            saw_compile_only_constructor);
        fflush(stderr);
    }
    return saw_cache_key_constructor && saw_compile_only_constructor;
}

NRT_STATUS nrt_get_instance_info(
    nrt_instance_info_t *info,
    size_t instance_info_len) {
    NRT_STATUS status;
    const char *target_override;

    if (pthread_once(&resolve_once, resolve_nrt_get_instance_info) != 0 ||
        real_nrt_get_instance_info == NULL) {
        return NRT_FAILURE;
    }

    status = real_nrt_get_instance_info(info, instance_info_len);
    target_override = getenv("QWEN35_CACHE_PLATFORM_TARGET");
    if (status != NRT_SUCCESS || info == NULL || target_override == NULL ||
        !called_from_cache_key_constructor()) {
        return status;
    }

    int original_family = (int)info->family;
    if (strcmp(target_override, "trn1") == 0) {
        info->family = NRT_INSTANCE_TRN1;
    } else if (strcmp(target_override, "trn2") == 0) {
        info->family = NRT_INSTANCE_TRN2;
    }
    const char *debug = getenv("QWEN35_PLATFORM_TARGET_SHIM_DEBUG");
    if (debug != NULL && strcmp(debug, "0") != 0) {
        fprintf(
            stderr,
            "qwen35-platform-override pid=%ld cache_key=1 original_family=%d "
            "target=%s overridden_family=%d\n",
            (long)getpid(),
            original_family,
            target_override,
            (int)info->family);
        fflush(stderr);
    }

    return status;
}
