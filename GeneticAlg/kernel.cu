#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <chrono>
#include <thrust/host_vector.h>
#include <thrust/extrema.h>
#include <fstream>

__device__ float evaluateFitness(const float* coefficients, const float* pointsX, const float* pointsY, int degree, int numPoints) {
    float mse = 0.0f;
    for (int i = 0; i < numPoints; ++i) {
        float predictedY = 0.0f;
        for (int j = 0; j <= degree; ++j) {
            predictedY += coefficients[j] * powf(pointsX[i], j);
        }
        float error = predictedY - pointsY[i];
        mse += error * error;
    }
    return mse / numPoints;
}

__device__ int tournamentSelection(float* fitness, int populationSize, curandState* state) {
    int bestIndividual = -1;
    float bestFitness = FLT_MAX;
    for (int i = 0; i < 10; ++i) {
        int individual = curand(state) % populationSize;
        if (fitness[individual] < bestFitness) {
            bestFitness = fitness[individual];
            bestIndividual = individual;
        }
    }
    return bestIndividual;
}

__device__ void crossover(const float* parent1, const float* parent2, float* child, int degree, curandState* state) {
    int crossoverPoint = 1 + (curand(state) % degree);
    for (int i = 0; i < degree + 1; ++i) {
        if (i < crossoverPoint) child[i] = parent1[i];
        else  child[i] = parent2[i];
    }
}

__device__ void mutate(float* individual, int degree, curandState* state) {
    for (int i = 0; i <= degree; ++i) individual[i] += curand_uniform(state) - 0.5f;
}

__global__ void geneticAlgorithmKernel(float* population, float* pointsX, float* pointsY, float* fitness, int populationSize, int degree, int numPoints, curandState* states) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < populationSize) {
        curand_init(1234, idx, 0, &states[idx]);
        fitness[idx] = evaluateFitness(&population[idx * (degree + 1)], pointsX, pointsY, degree, numPoints);
    }
}

__global__ void nextGeneration(float* population, float* newPopulation, float* fitness, int populationSize, int degree, bool applyMutation, curandState* states) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < populationSize) {
        curandState localState = states[idx];
        int parent1Idx = tournamentSelection(fitness, populationSize, &localState);
        int parent2Idx = tournamentSelection(fitness, populationSize, &localState);
        float* parent1 = &population[parent1Idx * (degree + 1)];
        float* parent2 = &population[parent2Idx * (degree + 1)];

        float* child = &newPopulation[idx * (degree + 1)];
        crossover(parent1, parent2, child, degree, &localState);
        if (applyMutation) {
            mutate(child, degree, &localState);
        }
    }
}

int main() {
    setlocale(LC_ALL, "RU");
    int degree = 4;
    int numPoints = 500;
    int populationSize = 1000;
    int maxGenerations = 1000;

    std::vector<float> hostPointsX(numPoints);
    std::vector<float> hostPointsY(numPoints);
    for (int i = 0; i < numPoints; ++i) {
        hostPointsX[i] = static_cast<float>((i + 1) / 50);
        hostPointsY[i] = 1 * hostPointsX[i] * hostPointsX[i] * hostPointsX[i] * hostPointsX[i] +
            2 * hostPointsX[i] * hostPointsX[i] * hostPointsX[i] +
            3 * hostPointsX[i] * hostPointsX[i] +
            4 * hostPointsX[i] +
            5;
    }
    std::vector<float> hostPopulation(populationSize * (degree + 1));
    for (int i = 0; i < populationSize * (degree + 1); ++i) {
        hostPopulation[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    float* devicePopulation, * devicePointsX, * devicePointsY, * deviceFitness, * deviceNewPopulation;
    cudaMalloc(&devicePopulation, populationSize * (degree + 1) * sizeof(float));
    cudaMalloc(&deviceNewPopulation, populationSize * (degree + 1) * sizeof(float));
    cudaMalloc(&devicePointsX, numPoints * sizeof(float));
    cudaMalloc(&devicePointsY, numPoints * sizeof(float));
    cudaMalloc(&deviceFitness, populationSize * sizeof(float));
    cudaMemcpy(devicePopulation, hostPopulation.data(), populationSize * (degree + 1) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devicePointsX, hostPointsX.data(), numPoints * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devicePointsY, hostPointsY.data(), numPoints * sizeof(float), cudaMemcpyHostToDevice);

    curandState* deviceStates;
    cudaMalloc(&deviceStates, populationSize * sizeof(curandState));

    float bestFitness = 1e16f;
    std::vector<float> bestCoefficients(degree + 1);
    float previousBestFitness = 1e16f;
    int generation = 0;
    int repeatCount = 0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (generation; generation < maxGenerations; ++generation) {
        geneticAlgorithmKernel << <populationSize / 1024+1, 1024 >> > (devicePopulation, devicePointsX, devicePointsY, deviceFitness, populationSize, degree, numPoints, deviceStates);
        cudaDeviceSynchronize();
        bool applyMutation = (repeatCount >= 4);

        nextGeneration << <populationSize/1024+1, 1024 >> > (devicePopulation, deviceNewPopulation, deviceFitness, populationSize, degree, applyMutation, deviceStates);
        cudaDeviceSynchronize();
        cudaMemcpy(hostPopulation.data(), deviceNewPopulation, populationSize * (degree + 1) * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(devicePopulation, deviceNewPopulation, populationSize * (degree + 1) * sizeof(float), cudaMemcpyHostToDevice);

        thrust::device_ptr<float> dev_ptr(deviceFitness);
        thrust::device_ptr<float> min_ptr = thrust::min_element(dev_ptr, dev_ptr + populationSize);
        bestFitness = *min_ptr;

        int bestIndex = min_ptr - dev_ptr;
        std::copy(hostPopulation.begin() + bestIndex * (degree + 1), hostPopulation.begin() + (bestIndex + 1) * (degree + 1), bestCoefficients.begin());

        if (bestFitness == previousBestFitness) repeatCount++;
        else repeatCount = 0;
        previousBestFitness = bestFitness;
        if (bestFitness < 0.1) break;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Время выполнения на GPU: " << milliseconds / 1000.0f << " секунд" << std::endl;
    std::cout << "Лучшие коэффициенты полинома: ";
    for (float coeff : bestCoefficients) {
        std::cout << coeff << " ";
    }
    std::cout << std::endl;
    std::cout << "Лучшая приспособленность: " << bestFitness << std::endl;
    std::cout << "Количество итераций: " << generation << std::endl;
    std::ofstream outFile("results.csv");
    outFile << "X,Y,PredictedY\n";
    for (int i = 0; i < numPoints; ++i) {
        float predictedY = 0.0f;
        for (int j = 0; j <= degree; ++j) {
            predictedY += bestCoefficients[j] * powf(hostPointsX[i], j);
        }
        outFile << hostPointsX[i] << "," << hostPointsY[i] << "," << predictedY << "\n";
    }
    outFile.close();
    cudaFree(devicePopulation);
    cudaFree(deviceNewPopulation);
    cudaFree(devicePointsX);
    cudaFree(devicePointsY);
    cudaFree(deviceFitness);
    cudaFree(deviceStates);
    return 0;
}
