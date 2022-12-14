// Copyright 2009, Andrew Corrigan, acorriga@gmu.edu
// This code is from the AIAA-2009-4001 paper

#include <iostream>
#include <fstream>
#include <cmath>
#include <omp.h>

#ifndef block_length
#define block_length 32
#endif

/*
 * Options
 *
 */
#define GAMMA 1.4
#define iterations 2000

#define NDIM 3
#define NNB 4

#define RK 3 // 3rd order RK
#define ff_mach 1.2
#define deg_angle_of_attack 0.0f

/*
 * not options
 */
#define VAR_DENSITY 0
#define VAR_MOMENTUM 1
#define VAR_DENSITY_ENERGY (VAR_MOMENTUM + NDIM)
#define NVAR (VAR_DENSITY_ENERGY + 1)

#ifdef restrict
#define __restrict restrict
#else
#define __restrict
#endif


#ifndef BLOCK_SIZE
#define BLOCK_SIZE (32)
#endif

#include <cuda_runtime.h>
#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
static inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

/*
 * Generic functions
 */
template <typename T>
T *alloc(int N)
{
	return new T[N];
}

template <typename T>
void dealloc(T *array)
{
	delete[] array;
}

template <typename T>
void copy(T *dst, T *src, int N)
{
	for (int i = 0; i < N; i++)
	{
		dst[i] = src[i];
	}
}

void dump(float *variables, int nel, int nelr)
{
	{
		std::ofstream file("density");
		file << nel << " " << nelr << std::endl;
		for (int i = 0; i < nel; i++)
			file << variables[i + VAR_DENSITY * nelr] << std::endl;
	}

	{
		std::ofstream file("momentum");
		file << nel << " " << nelr << std::endl;
		for (int i = 0; i < nel; i++)
		{
			for (int j = 0; j != NDIM; j++)
				file << variables[i + (VAR_MOMENTUM + j) * nelr] << " ";
			file << std::endl;
		}
	}

	{
		std::ofstream file("density_energy");
		file << nel << " " << nelr << std::endl;
		for (int i = 0; i < nel; i++)
			file << variables[i + VAR_DENSITY_ENERGY * nelr] << std::endl;
	}
}

/*Creazione costanti che utilizza cuda per fare i diversi calcoli*/
__constant__ float ff_variable[NVAR];
__constant__ float3 ff_flux_contribution_momentum_x;
__constant__ float3 ff_flux_contribution_momentum_y;
__constant__ float3 ff_flux_contribution_momentum_z;
__constant__ float3 ff_flux_contribution_density_energy;


//funzione per inizializzare le variabili
__global__ void cuda_initialize_variables(int nelr, float* variables)
{
	/* Original code of this function
	for (int i = 0; i < nelr; i++)
	{
		for (int j = 0; j < NVAR; j++)
			variables[i + j * nelr] = ff_variable[j];
	}
	*/
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	for (int j = 0; j < NVAR; j++)
		variables[i + j * nelr] = ff_variable[j];
}

void initialize_variables(int nelr, float *variables)
{
	dim3 dimGrid(nelr / BLOCK_SIZE),    dimBlock(BLOCK_SIZE);
	cuda_initialize_variables<<<dimGrid, dimBlock>>>(nelr, variables);
	gpuErrchk(cudaPeekAtLastError());

}
// questa funzione ?? eseguita sull'host ed ?? chiamata sia dal device che dall'host
__device__ __host__ inline void compute_flux_contribution(float &density, float3 &momentum, float &density_energy, float &pressure, float3 &velocity, float3 &fc_momentum_x, float3 &fc_momentum_y, float3 &fc_momentum_z, float3 &fc_density_energy)
{
	fc_momentum_x.x = velocity.x * momentum.x + pressure;
	fc_momentum_x.y = velocity.x * momentum.y;
	fc_momentum_x.z = velocity.x * momentum.z;

	fc_momentum_y.x = fc_momentum_x.y;
	fc_momentum_y.y = velocity.y * momentum.y + pressure;
	fc_momentum_y.z = velocity.y * momentum.z;

	fc_momentum_z.x = fc_momentum_x.z;
	fc_momentum_z.y = fc_momentum_y.z;
	fc_momentum_z.z = velocity.z * momentum.z + pressure;

	float de_p = density_energy + pressure;
	fc_density_energy.x = velocity.x * de_p;
	fc_density_energy.y = velocity.y * de_p;
	fc_density_energy.z = velocity.z * de_p;
}

// funzione eseguita dal device e chiamabile solamente dal device
__device__ inline void compute_velocity(float &density, float3 &momentum, float3 &velocity)
{
	velocity.x = momentum.x / density;
	velocity.y = momentum.y / density;
	velocity.z = momentum.z / density;
}
// funzione eseguita dal device e chiamabile solamente dal device
__device__ inline float compute_speed_sqd(float3 &velocity)
{
	return velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z;
}
// funzione eseguita dal device e chiamabile solamente dal device
__device__ inline float compute_pressure(float &density, float &density_energy, float &speed_sqd)
{
	return (float(GAMMA) - float(1.0f)) * (density_energy - float(0.5f) * density * speed_sqd);
}
// funzione eseguita dal device e chiamabile solamente dal device
__device__ inline float compute_speed_of_sound(float &density, float &pressure)
{
	return std::sqrt(float(GAMMA) * pressure / density);
}
//funzione eseguita dal device e chiamabile solo dall'host
__global__ void compute_step_factor(int nelr, float *__restrict variables, float *areas, float *__restrict step_factors)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	//for (int blk = 0; blk < nelr / block_length; ++blk)
	//{
		//int b_start = blk * block_length;
		//int b_end = (blk + 1) * block_length > nelr ? nelr : (blk + 1) * block_length;
		//for (int i = b_start; i < b_end; i++)
		//{
			float density = variables[i + VAR_DENSITY * nelr];

			float3 momentum;
			momentum.x = variables[i + (VAR_MOMENTUM + 0) * nelr];
			momentum.y = variables[i + (VAR_MOMENTUM + 1) * nelr];
			momentum.z = variables[i + (VAR_MOMENTUM + 2) * nelr];

			float density_energy = variables[i + VAR_DENSITY_ENERGY * nelr];
			float3 velocity;
			compute_velocity(density, momentum, velocity);
			float speed_sqd = compute_speed_sqd(velocity);
			float pressure = compute_pressure(density, density_energy, speed_sqd);
			float speed_of_sound = compute_speed_of_sound(density, pressure);

			// dt = float(0.5f) * std::sqrt(areas[i]) /  (||v|| + c).... but when we do time stepping, this later would need to be divided by the area, so we just do it all at once
			step_factors[i] = float(0.5f) / (std::sqrt(areas[i]) * (std::sqrt(speed_sqd) + speed_of_sound));
		//}
	//}
}

/*
 *
 *
 */
//funzione eseguita dal device e chiamabile solo dall'host
__global__ void compute_flux(int nelr, int *elements_surrounding_elements, float *normals, float *variables, float *fluxes)
{
	const float smoothing_coefficient = float(0.2f);
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j, nb;
	// for (int blk = 0; blk < nelr / block_length; ++blk)
	// {
	// 	int b_start = blk * block_length;
	// 	int b_end = (blk + 1) * block_length > nelr ? nelr : (blk + 1) * block_length;
	// 	for (int i = b_start; i < b_end; ++i)
	// 	{
			float density_i = variables[i + VAR_DENSITY * nelr];
			float3 momentum_i;
			momentum_i.x = variables[i + (VAR_MOMENTUM + 0) * nelr];
			momentum_i.y = variables[i + (VAR_MOMENTUM + 1) * nelr];
			momentum_i.z = variables[i + (VAR_MOMENTUM + 2) * nelr];

			float density_energy_i = variables[i + VAR_DENSITY_ENERGY * nelr];

			float3 velocity_i;
			compute_velocity(density_i, momentum_i, velocity_i);
			float speed_sqd_i = compute_speed_sqd(velocity_i);
			float speed_i = std::sqrt(speed_sqd_i);
			float pressure_i = compute_pressure(density_i, density_energy_i, speed_sqd_i);
			float speed_of_sound_i = compute_speed_of_sound(density_i, pressure_i);
			float3 flux_contribution_i_momentum_x, flux_contribution_i_momentum_y, flux_contribution_i_momentum_z;
			float3 flux_contribution_i_density_energy;
			compute_flux_contribution(density_i, momentum_i, density_energy_i, pressure_i, velocity_i, flux_contribution_i_momentum_x, flux_contribution_i_momentum_y, flux_contribution_i_momentum_z, flux_contribution_i_density_energy);

			float flux_i_density = float(0.0f);
			float3 flux_i_momentum;
			flux_i_momentum.x = float(0.0f);
			flux_i_momentum.y = float(0.0f);
			flux_i_momentum.z = float(0.0f);
			float flux_i_density_energy = float(0.0f);

			float3 velocity_nb;
			float density_nb, density_energy_nb;
			float3 momentum_nb;
			float3 flux_contribution_nb_momentum_x, flux_contribution_nb_momentum_y, flux_contribution_nb_momentum_z;
			float3 flux_contribution_nb_density_energy;
			float speed_sqd_nb, speed_of_sound_nb, pressure_nb;
		//FORSE SI PUO' PARALLELIZZARE ANCHE QUESTO CON CUDA
#pragma unroll
			for (j = 0; j < NNB; j++)
			{
				float3 normal;
				float normal_len;
				float factor;

				nb = elements_surrounding_elements[i + j * nelr];
				normal.x = normals[i + (j + 0 * NNB) * nelr];
				normal.y = normals[i + (j + 1 * NNB) * nelr];
				normal.z = normals[i + (j + 2 * NNB) * nelr];
				normal_len = std::sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);

				if (nb >= 0) // a legitimate neighbor
				{
					density_nb = variables[nb + VAR_DENSITY * nelr];
					momentum_nb.x = variables[nb + (VAR_MOMENTUM + 0) * nelr];
					momentum_nb.y = variables[nb + (VAR_MOMENTUM + 1) * nelr];
					momentum_nb.z = variables[nb + (VAR_MOMENTUM + 2) * nelr];
					density_energy_nb = variables[nb + VAR_DENSITY_ENERGY * nelr];
					compute_velocity(density_nb, momentum_nb, velocity_nb);
					speed_sqd_nb = compute_speed_sqd(velocity_nb);
					pressure_nb = compute_pressure(density_nb, density_energy_nb, speed_sqd_nb);
					speed_of_sound_nb = compute_speed_of_sound(density_nb, pressure_nb);
					compute_flux_contribution(density_nb, momentum_nb, density_energy_nb, pressure_nb, velocity_nb, flux_contribution_nb_momentum_x, flux_contribution_nb_momentum_y, flux_contribution_nb_momentum_z, flux_contribution_nb_density_energy);

					// artificial viscosity
					factor = -normal_len * smoothing_coefficient * float(0.5f) * (speed_i + std::sqrt(speed_sqd_nb) + speed_of_sound_i + speed_of_sound_nb);
					flux_i_density += factor * (density_i - density_nb);
					flux_i_density_energy += factor * (density_energy_i - density_energy_nb);
					flux_i_momentum.x += factor * (momentum_i.x - momentum_nb.x);
					flux_i_momentum.y += factor * (momentum_i.y - momentum_nb.y);
					flux_i_momentum.z += factor * (momentum_i.z - momentum_nb.z);

					// accumulate cell-centered fluxes
					factor = float(0.5f) * normal.x;
					flux_i_density += factor * (momentum_nb.x + momentum_i.x);
					flux_i_density_energy += factor * (flux_contribution_nb_density_energy.x + flux_contribution_i_density_energy.x);
					flux_i_momentum.x += factor * (flux_contribution_nb_momentum_x.x + flux_contribution_i_momentum_x.x);
					flux_i_momentum.y += factor * (flux_contribution_nb_momentum_y.x + flux_contribution_i_momentum_y.x);
					flux_i_momentum.z += factor * (flux_contribution_nb_momentum_z.x + flux_contribution_i_momentum_z.x);

					factor = float(0.5f) * normal.y;
					flux_i_density += factor * (momentum_nb.y + momentum_i.y);
					flux_i_density_energy += factor * (flux_contribution_nb_density_energy.y + flux_contribution_i_density_energy.y);
					flux_i_momentum.x += factor * (flux_contribution_nb_momentum_x.y + flux_contribution_i_momentum_x.y);
					flux_i_momentum.y += factor * (flux_contribution_nb_momentum_y.y + flux_contribution_i_momentum_y.y);
					flux_i_momentum.z += factor * (flux_contribution_nb_momentum_z.y + flux_contribution_i_momentum_z.y);

					factor = float(0.5f) * normal.z;
					flux_i_density += factor * (momentum_nb.z + momentum_i.z);
					flux_i_density_energy += factor * (flux_contribution_nb_density_energy.z + flux_contribution_i_density_energy.z);
					flux_i_momentum.x += factor * (flux_contribution_nb_momentum_x.z + flux_contribution_i_momentum_x.z);
					flux_i_momentum.y += factor * (flux_contribution_nb_momentum_y.z + flux_contribution_i_momentum_y.z);
					flux_i_momentum.z += factor * (flux_contribution_nb_momentum_z.z + flux_contribution_i_momentum_z.z);
				}
				else if (nb == -1) // a wing boundary
				{
					flux_i_momentum.x += normal.x * pressure_i;
					flux_i_momentum.y += normal.y * pressure_i;
					flux_i_momentum.z += normal.z * pressure_i;
				}
				else if (nb == -2) // a far field boundary
				{
					factor = float(0.5f) * normal.x;
					flux_i_density += factor * (ff_variable[VAR_MOMENTUM + 0] + momentum_i.x);
					flux_i_density_energy += factor * (ff_flux_contribution_density_energy.x + flux_contribution_i_density_energy.x);
					flux_i_momentum.x += factor * (ff_flux_contribution_momentum_x.x + flux_contribution_i_momentum_x.x);
					flux_i_momentum.y += factor * (ff_flux_contribution_momentum_y.x + flux_contribution_i_momentum_y.x);
					flux_i_momentum.z += factor * (ff_flux_contribution_momentum_z.x + flux_contribution_i_momentum_z.x);

					factor = float(0.5f) * normal.y;
					flux_i_density += factor * (ff_variable[VAR_MOMENTUM + 1] + momentum_i.y);
					flux_i_density_energy += factor * (ff_flux_contribution_density_energy.y + flux_contribution_i_density_energy.y);
					flux_i_momentum.x += factor * (ff_flux_contribution_momentum_x.y + flux_contribution_i_momentum_x.y);
					flux_i_momentum.y += factor * (ff_flux_contribution_momentum_y.y + flux_contribution_i_momentum_y.y);
					flux_i_momentum.z += factor * (ff_flux_contribution_momentum_z.y + flux_contribution_i_momentum_z.y);

					factor = float(0.5f) * normal.z;
					flux_i_density += factor * (ff_variable[VAR_MOMENTUM + 2] + momentum_i.z);
					flux_i_density_energy += factor * (ff_flux_contribution_density_energy.z + flux_contribution_i_density_energy.z);
					flux_i_momentum.x += factor * (ff_flux_contribution_momentum_x.z + flux_contribution_i_momentum_x.z);
					flux_i_momentum.y += factor * (ff_flux_contribution_momentum_y.z + flux_contribution_i_momentum_y.z);
					flux_i_momentum.z += factor * (ff_flux_contribution_momentum_z.z + flux_contribution_i_momentum_z.z);
				}
			}
			fluxes[i + VAR_DENSITY * nelr] = flux_i_density;
			fluxes[i + (VAR_MOMENTUM + 0) * nelr] = flux_i_momentum.x;
			fluxes[i + (VAR_MOMENTUM + 1) * nelr] = flux_i_momentum.y;
			fluxes[i + (VAR_MOMENTUM + 2) * nelr] = flux_i_momentum.z;
			fluxes[i + VAR_DENSITY_ENERGY * nelr] = flux_i_density_energy;
		//}
	//}
}

__global__ void time_step(int j, int nelr, float *old_variables, float *variables, float *step_factors, float *fluxes)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	// for (int blk = 0; blk < nelr / block_length; ++blk)
	// {
	// 	int b_start = blk * block_length;
	// 	int b_end = (blk + 1) * block_length > nelr ? nelr : (blk + 1) * block_length;
	// 	for (int i = b_start; i < b_end; ++i)
	// 	{
			float factor = step_factors[i] / float(RK + 1 - j);

			variables[i + VAR_DENSITY * nelr] = old_variables[i + VAR_DENSITY * nelr] + factor * fluxes[i + VAR_DENSITY * nelr];
			variables[i + (VAR_MOMENTUM + 0) * nelr] = old_variables[i + (VAR_MOMENTUM + 0) * nelr] + factor * fluxes[i + (VAR_MOMENTUM + 0) * nelr];
			variables[i + (VAR_MOMENTUM + 1) * nelr] = old_variables[i + (VAR_MOMENTUM + 1) * nelr] + factor * fluxes[i + (VAR_MOMENTUM + 1) * nelr];
			variables[i + (VAR_MOMENTUM + 2) * nelr] = old_variables[i + (VAR_MOMENTUM + 2) * nelr] + factor * fluxes[i + (VAR_MOMENTUM + 2) * nelr];
			variables[i + VAR_DENSITY_ENERGY * nelr] = old_variables[i + VAR_DENSITY_ENERGY * nelr] + factor * fluxes[i + VAR_DENSITY_ENERGY * nelr];
		//}
	//}
}
/*
 * Main function
 */
int main(int argc, char **argv)
{
	if (argc < 2)
	{
		std::cout << "specify data file name" << std::endl;
		return 0;
	}
	const char *data_file_name = argv[1];

	float h_ff_variable[NVAR];
	float3 h_ff_flux_contribution_momentum_x, h_ff_flux_contribution_momentum_y, h_ff_flux_contribution_momentum_z, h_ff_flux_contribution_density_energy;

	// set far field conditions
	{
		const float angle_of_attack = float(3.1415926535897931 / 180.0f) * float(deg_angle_of_attack);

		h_ff_variable[VAR_DENSITY] = float(1.4);

		float ff_pressure = float(1.0f);
		float ff_speed_of_sound = sqrt(GAMMA * ff_pressure / h_ff_variable[VAR_DENSITY]);
		float ff_speed = float(ff_mach) * ff_speed_of_sound;

		float3 ff_velocity;
		ff_velocity.x = ff_speed * float(cos((float)angle_of_attack));
		ff_velocity.y = ff_speed * float(sin((float)angle_of_attack));
		ff_velocity.z = 0.0f;

		h_ff_variable[VAR_MOMENTUM + 0] = h_ff_variable[VAR_DENSITY] * ff_velocity.x;
		h_ff_variable[VAR_MOMENTUM + 1] = h_ff_variable[VAR_DENSITY] * ff_velocity.y;
		h_ff_variable[VAR_MOMENTUM + 2] = h_ff_variable[VAR_DENSITY] * ff_velocity.z;

		h_ff_variable[VAR_DENSITY_ENERGY] = h_ff_variable[VAR_DENSITY] * (float(0.5f) * (ff_speed * ff_speed)) + (ff_pressure / float(GAMMA - 1.0f));

		float3 h_ff_momentum;
		h_ff_momentum.x = *(h_ff_variable + VAR_MOMENTUM + 0);
		h_ff_momentum.y = *(h_ff_variable + VAR_MOMENTUM + 1);
		h_ff_momentum.z = *(h_ff_variable + VAR_MOMENTUM + 2);
		compute_flux_contribution(h_ff_variable[VAR_DENSITY], h_ff_momentum, h_ff_variable[VAR_DENSITY_ENERGY], ff_pressure, ff_velocity, h_ff_flux_contribution_momentum_x, h_ff_flux_contribution_momentum_y, h_ff_flux_contribution_momentum_z, h_ff_flux_contribution_density_energy);

		//alloco memoria per la GPU
		// copy far field conditions to the gpu
		gpuErrchk(cudaMemcpyToSymbol(ff_variable,          h_ff_variable,          NVAR*sizeof(float)));
		gpuErrchk(cudaMemcpyToSymbol(ff_flux_contribution_momentum_x, &h_ff_flux_contribution_momentum_x, sizeof(float3)));
		gpuErrchk(cudaMemcpyToSymbol(ff_flux_contribution_momentum_y, &h_ff_flux_contribution_momentum_y, sizeof(float3)));
		gpuErrchk(cudaMemcpyToSymbol(ff_flux_contribution_momentum_z, &h_ff_flux_contribution_momentum_z, sizeof(float3)));

		gpuErrchk(cudaMemcpyToSymbol(ff_flux_contribution_density_energy, &h_ff_flux_contribution_density_energy, sizeof(float3)));

	}
	int nel;
	int nelr;

	// read in domain geometry
	float *areas;
	int *elements_surrounding_elements;
	float *normals;
	{
		std::ifstream file(data_file_name);

		file >> nel;
		nelr = block_length * ((nel / block_length) + std::min(1, nel % block_length));

		areas = new float[nelr];
		elements_surrounding_elements = new int[nelr * NNB];
		normals = new float[NDIM * NNB * nelr];

		// read in data
		for (int i = 0; i < nel; i++)
		{
			file >> areas[i];
			for (int j = 0; j < NNB; j++)
			{
				file >> elements_surrounding_elements[i + j * nelr];
				if (elements_surrounding_elements[i + j * nelr] < 0)
					elements_surrounding_elements[i + j * nelr] = -1;
				elements_surrounding_elements[i + j * nelr]--; // it's coming in with Fortran numbering

				for (int k = 0; k < NDIM; k++)
				{
					file >> normals[i + (j + k * NNB) * nelr];
					normals[i + (j + k * NNB) * nelr] = -normals[i + (j + k * NNB) * nelr];
				}
			}
		}

		// fill in remaining data
		int last = nel - 1;
		for (int i = nel; i < nelr; i++)
		{
			areas[i] = areas[last];
			for (int j = 0; j < NNB; j++)
			{
				// duplicate the last element
				elements_surrounding_elements[i + j * nelr] = elements_surrounding_elements[last + j * nelr];
				for (int k = 0; k < NDIM; k++)
					normals[i + (j + k * NNB) * nelr] = normals[last + (j + k * NNB) * nelr];
			}
		}
	}

	// Create arrays and set initial conditions
	float *variables = alloc<float>(nelr * NVAR);
	float *d_variables;
	gpuErrchk(cudaMalloc((void **)&d_variables, sizeof(float) * nelr * NVAR));
	initialize_variables(nelr, d_variables);

	float *d_old_variables;
	gpuErrchk(cudaMalloc((void **)&d_old_variables, sizeof(float) * nelr * NVAR));
	float *d_fluxes;
	gpuErrchk(cudaMalloc((void **)&d_fluxes, sizeof(float) * nelr * NVAR));
	float *d_step_factors;
	gpuErrchk(cudaMalloc((void **)&d_step_factors, sizeof(float) * nelr));


	float *d_areas, *d_normals;
	int *d_elements_surrounding_elements;
	gpuErrchk(cudaMalloc((void **)&d_areas, sizeof(float)*nelr));
	gpuErrchk(cudaMalloc((void **)&d_elements_surrounding_elements, sizeof(int)*nelr*NNB));
	gpuErrchk(cudaMalloc((void **)&d_normals, sizeof(float)*NDIM * NNB * nelr));
	gpuErrchk(cudaMemcpy(d_areas, areas, sizeof(float)*nelr, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_elements_surrounding_elements, elements_surrounding_elements, sizeof(float)*nelr*NNB, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_normals, normals, sizeof(float)*NDIM * NNB * nelr, cudaMemcpyHostToDevice));

	//Chiamo il kernel per inizializzare le variabili
	//initialize_variables(nelr, old_variables, ff_variable);
	//initialize_variables(nelr, fluxes, ff_variable);

	// these need to be computed the first time in order to compute time step
	std::cout << "Starting..." << std::endl;

	// Begin iterations
	for (int i = 0; i < 2000; i++)
	{
		std::cout << i << "/" << iterations << std::endl;
		{
		gpuErrchk(cudaMemcpy(d_old_variables, d_variables, sizeof(float) * nelr * NVAR, cudaMemcpyDeviceToDevice));


		// for the first iteration we compute the time step
		//compute_step_factor(nelr, variables, areas, step_factors);
		//dim3 gridDim(nelr / BLOCK_SIZE, nelr/BLOCK_SIZE), blockDim(BLOCK_SIZE, BLOCK_SIZE);
		int blockDim = (nelr + BLOCK_SIZE - 1) / BLOCK_SIZE;
		compute_step_factor<<<blockDim, BLOCK_SIZE>>>(nelr, d_variables, d_areas, d_step_factors);
		gpuErrchk(cudaPeekAtLastError());

		for (int j = 0; j < RK; j++)
		{
			//compute_flux(nelr, elements_surrounding_elements, normals, variables, fluxes, ff_variable, ff_flux_contribution_momentum_x, ff_flux_contribution_momentum_y, ff_flux_contribution_momentum_z, ff_flux_contribution_density_energy);
			compute_flux<<<blockDim, BLOCK_SIZE>>>(nelr, d_elements_surrounding_elements, d_normals, d_variables, d_fluxes);
			gpuErrchk(cudaPeekAtLastError());
			//time_step(j, nelr, old_variables, variables, step_factors, fluxes);
			time_step<<<blockDim, BLOCK_SIZE>>>(j, nelr, d_old_variables, d_variables, d_step_factors, d_fluxes);
			gpuErrchk(cudaPeekAtLastError());
		}
		}
	}

	gpuErrchk(cudaMemcpy(variables, d_variables, sizeof(float) * nelr * NVAR, cudaMemcpyDeviceToHost));

	std::cout << "Saving solution..." << std::endl;
	dump(variables, nel, nelr);
	std::cout << "Saved solution..." << std::endl;

	std::cout << "Cleaning up..." << std::endl;
	dealloc<float>(areas);
	dealloc<int>(elements_surrounding_elements);
	dealloc<float>(normals);

	dealloc<float>(variables);

	cudaFree(d_variables);
	cudaFree(d_old_variables);
	cudaFree(d_step_factors);
	cudaFree(d_fluxes);
	cudaFree(d_areas);
	cudaFree(d_elements_surrounding_elements);
	cudaFree(d_normals);

	std::cout << "Done..." << std::endl;

	return 0;
}
