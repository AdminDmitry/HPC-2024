#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <fstream>

#define MAX_DEPTH 5

inline cudaError_t checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result) << std::endl;
        exit(-1);
    }
    return result;
}

struct Vec3 {
    float x, y, z;
    __host__ __device__ Vec3 operator+(const Vec3& v) const { return {x + v.x, y + v.y, z + v.z}; }
    __host__ __device__ Vec3 operator-(const Vec3& v) const { return {x - v.x, y - v.y, z - v.z}; }
    __host__ __device__ Vec3 operator*(float scalar) const { return {x * scalar, y * scalar, z * scalar}; }
    __host__ __device__ float dot(const Vec3& v) const { return x * v.x + y * v.y + z * v.z; }
    __host__ __device__ Vec3 normalize() const {
        float len = sqrtf(dot(*this));
        return (len > 0) ? *this * (1.0f / len) : Vec3{0, 0, 0};
    }
};

struct Sphere {
    Vec3 center;
    float radius;
    Vec3 color;
};

struct Light {
    Vec3 position;
    Vec3 intensity;
};

struct Plane {
    Vec3 point;
    Vec3 normal;
    Vec3 color;
};

__device__ bool intersectSphere(const Vec3& rayOrigin, const Vec3& rayDir, const Sphere& sphere, float& t) {
    Vec3 oc = rayOrigin - sphere.center;
    float a = rayDir.dot(rayDir);
    float b = 2.0f * oc.dot(rayDir);
    float c = oc.dot(oc) - sphere.radius * sphere.radius;
    float discriminant = b * b - 4 * a * c;
    if (discriminant < 0) return false;
    t = (-b - sqrtf(discriminant)) / (2.0f * a);
    return t >= 0;
}

__device__ bool intersectPlane(const Vec3& rayOrigin, const Vec3& rayDir, const Plane& plane, float& t) {
    float denom = plane.normal.dot(rayDir);
    if (fabs(denom) > 1e-6) {
        Vec3 p0l0 = plane.point - rayOrigin;
        t = p0l0.dot(plane.normal) / denom;
        return t >= 0;
    }
    return false;
}

__device__ Vec3 calculateLighting(const Vec3& intersection, const Vec3& normal, const Vec3& viewDir, const Vec3& objectColor, Light* lights, int numLights, bool inShadow) {
    constexpr float AMBIENT_COEFFICIENT = 0.1f;
    constexpr float DIFFUSE_COEFFICIENT = 1.0f;
    constexpr float SPECULAR_COEFFICIENT = 0.7f;
    constexpr float SHININESS = 32.0f;

    Vec3 color = objectColor * AMBIENT_COEFFICIENT;

    for (int i = 0; i < numLights; ++i) {
        Vec3 lightDir = (lights[i].position - intersection).normalize();
        if (inShadow) continue;

        // Diffuse
        float diff = fmaxf(normal.dot(lightDir), 0.0f);
        color = color + objectColor * (lights[i].intensity * DIFFUSE_COEFFICIENT * diff);

        // Specular
        Vec3 reflectDir = (normal * (2.0f * normal.dot(lightDir)) - lightDir).normalize();
        float spec = powf(fmaxf(viewDir.dot(reflectDir), 0.0f), SHININESS);
        color = color + lights[i].intensity * SPECULAR_COEFFICIENT * spec;
    }

    return color;
}

__device__ Vec3 TraceRay(const Vec3& rayOrigin, const Vec3& rayDir, Sphere* spheres, int numSpheres, Plane* planes, int numPlanes, Light* lights, int numLights, int depth) {
    if (depth > MAX_DEPTH) return {0.0f, 0.0f, 0.0f};

    float closestT = 1e20f;
    int closestSphere = -1, closestPlane = -1;
    bool hitPlane = false;

    for (int i = 0; i < numSpheres; ++i) {
        float t;
        if (intersectSphere(rayOrigin, rayDir, spheres[i], t) && t < closestT) {
            closestT = t;
            closestSphere = i;
            hitPlane = false;
        }
    }

    for (int i = 0; i < numPlanes; ++i) {
        float t;
        if (intersectPlane(rayOrigin, rayDir, planes[i], t) && t < closestT) {
            closestT = t;
            closestPlane = i;
            hitPlane = true;
        }
    }

    Vec3 intersectionPoint = rayOrigin + rayDir * closestT;
    Vec3 viewDir = -rayDir;
    Vec3 color = {0.0f, 0.0f, 0.0f};

    if (hitPlane && closestPlane != -1) {
        color = calculateLighting(intersectionPoint, planes[closestPlane].normal.normalize(), viewDir, planes[closestPlane].color, lights, numLights, false);
    } else if (closestSphere != -1) {
        Vec3 normal = (intersectionPoint - spheres[closestSphere].center).normalize();
        color = calculateLighting(intersectionPoint, normal, viewDir, spheres[closestSphere].color, lights, numLights, false);
    }

    return color;
}

__global__ void renderKernel(Sphere* spheres, int numSpheres, Plane* planes, int numPlanes, Light* lights, int numLights, unsigned char* image, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float u = float(x) / float(width);
        float v = float(y) / float(height);

        Vec3 rayOrigin = {0.0f, 0.0f, 0.0f};
        Vec3 rayDir = {u - 0.5f, v - 0.5f, -1.0f};
        rayDir = rayDir.normalize();

        Vec3 color = TraceRay(rayOrigin, rayDir, spheres, numSpheres, planes, numPlanes, lights, numLights, 0);

        int pixelIndex = (y * width + x) * 3;
        image[pixelIndex] = (unsigned char)(fminf(color.x, 1.0f) * 255);
        image[pixelIndex + 1] = (unsigned char)(fminf(color.y, 1.0f) * 255);
        image[pixelIndex + 2] = (unsigned char)(fminf(color.z, 1.0f) * 255);
    }
}

void renderScene(Sphere* spheres, int numSpheres, Plane* planes, int numPlanes, Light* lights, int numLights, unsigned char* image, int width, int height) {
    Sphere* d_spheres;
    Plane* d_planes;
    Light* d_lights;
    unsigned char* d_image;

    checkCuda(cudaMallocManaged(&d_spheres, sizeof(Sphere) * numSpheres));
    checkCuda(cudaMallocManaged(&d_planes, sizeof(Plane) * numPlanes));
    checkCuda(cudaMallocManaged(&d_lights, sizeof(Light) * numLights));
    checkCuda(cudaMallocManaged(&d_image, sizeof(unsigned char) * width * height * 3));

    cudaMemcpy(d_spheres, spheres, sizeof(Sphere) * numSpheres, cudaMemcpyHostToDevice);
    cudaMemcpy(d_planes, planes, sizeof(Plane) * numPlanes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_lights, lights, sizeof(Light) * numLights, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    renderKernel<<<gridSize, blockSize>>>(d_spheres, numSpheres, d_planes, numPlanes, d_lights, numLights, d_image, width, height);
    checkCuda(cudaDeviceSynchronize());

    cudaMemcpy(image, d_image, sizeof(unsigned char) * width * height * 3, cudaMemcpyDeviceToHost);

    cudaFree(d_spheres);
    cudaFree(d_planes);
    cudaFree(d_lights);
    cudaFree(d_image);
}

void saveBMP(const char* filename, unsigned char* image, int width, int height) {
    std::ofstream file(filename, std::ios::binary);

    unsigned char header[54] = {
        'B', 'M', 0, 0, 0, 0, 0, 0, 0, 0, 54, 0, 0, 0, 40, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };

    int fileSize = 54 + width * height * 3;
    header[2] = (unsigned char)(fileSize);
    header[3] = (unsigned char)(fileSize >> 8);
    header[4] = (unsigned char)(fileSize >> 16);
    header[5] = (unsigned char)(fileSize >> 24);

    header[18] = (unsigned char)(width);
    header[19] = (unsigned char)(width >> 8);
    header[20] = (unsigned char)(width >> 16);
    header[21] = (unsigned char)(width >> 24);

    header[22] = (unsigned char)(height);
    header[23] = (unsigned char)(height >> 8);
    header[24] = (unsigned char)(height >> 16);
    header[25] = (unsigned char)(height >> 24);

    file.write(reinterpret_cast<char*>(header), 54);
    file.write(reinterpret_cast<char*>(image), width * height * 3);
    file.close();
}

int main() {
    const int width = 4000;
    const int height = 4000;

    Sphere spheres[] = {
        {{-2.0f, 0.0f, -6.0f}, 1.0f, {1.0f, 0.0f, 0.0f}},
        {{0.0f, 0.0f, -8.0f}, 1.0f, {0.0f, 1.0f, 0.0f}},
        {{1.5f, 0.0f, -6.0f}, 1.0f, {0.0f, 0.0f, 1.0f}}
    };

    Plane planes[] = {
        {{0.0f, -1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f, 1.0f}},
        {{0.0f, 0.0f, -10.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f, 1.0f}}
    };

    Light lights[] = {
        {{-10.0f, 10.0f, 10.0f}, {1.0f, 1.0f, 1.0f}}
    };

    unsigned char* image = new unsigned char[width * height * 3];
    renderScene(spheres, 3, planes, 2, lights, 1, image, width, height);
    saveBMP("output.bmp", image, width, height);
    delete[] image;

    return 0;
}
