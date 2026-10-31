#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>

__attribute__((constructor)) static void pin_driver_modules() {
    const char * modules[] = {
        "libEGL_nvidia.so.0", "libGLX_nvidia.so.0", "libnvidia-gpucomp.so.550.163.01",
        "libnvidia-glsi.so.550.163.01", "libnvidia-glcore.so.550.163.01", "libnvidia-eglcore.so.550.163.01",
    };
    for (const char * name : modules) {
        if (!dlopen(name, RTLD_NOW | RTLD_NODELETE)) {
            std::fprintf(stderr, "driver pin failed: %s\n", name);
            std::abort();
        }
    }
}
