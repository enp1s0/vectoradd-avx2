#include <iostream>
#include <vector>
#include <chrono>
#include <immintrin.h>

using data_t = double;
constexpr std::size_t N = 1lu << 25;

template <unsigned n>
struct load_unroll {
	load_unroll(const double* const ptr, __m256d* const y) {
		load_unroll<n - 1>(ptr, y);
		y[n - 1] = _mm256_load_pd(ptr + (n - 1) * 4);
	}
};
template <>
struct load_unroll<0> {
	load_unroll(const double* const ptr, __m256d* const ymm) {}
};

template <unsigned n>
struct add_unroll {
	add_unroll(double* const ptr, const __m256d* const y, const __m256d* const x) {
		add_unroll<n - 1>(ptr, y, x);
		const auto z = _mm256_add_pd(x[n - 1], y[n - 1]);
		_mm256_store_pd(ptr + (n - 1) * 4, z);
	}
};
template <>
struct add_unroll<0> {
	add_unroll(double* const ptr, const __m256d* const y, const __m256d* const x) {}
};

template <class T, unsigned unrolling_len>
void vector_add(
	T* const z_ptr,
	const T* const x_ptr,
	const T* const y_ptr,
	const std::size_t len
	) {
	constexpr unsigned simd_len = 4;
#pragma omp parallel for
	for (std::size_t i = 0; i < len; i += unrolling_len * simd_len){
		__m256d x[unrolling_len];
		__m256d y[unrolling_len];
		load_unroll<unrolling_len>(x_ptr + i, x);
		load_unroll<unrolling_len>(y_ptr + i, y);
		add_unroll<unrolling_len>(z_ptr + i, x, y);
	}
}

template <class T, unsigned unrolling_len>
void eval_core(
	T* const z_ptr,
	const T* const x_ptr,
	const T* const y_ptr,
	const std::size_t len
	) {
	const auto start_clock = std::chrono::system_clock::now();
	vector_add<T, unrolling_len>(z_ptr, x_ptr, y_ptr, len);
	const auto end_clock = std::chrono::system_clock::now();
	const auto elapsed_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_clock - start_clock).count() * 1e-9;

	double max_relative_error = 0;
#pragma omp parallel for reduction(max: max_relative_error)
	for (std::size_t i = 0; i < N; i++) {
		const double correct = 2 * (i + 1);
		const auto diff = correct - z_ptr[i];
		max_relative_error = std::max(max_relative_error, std::abs(diff) / correct);
	}

	std::printf("unrolling len = %4u, Throughput : %e [GFlop/s], %e [GiB/s] (error=%e, %s)\n",
							unrolling_len,
							N / elapsed_time * 1e-9,
							3 * sizeof(data_t) * N / elapsed_time * 1e-9,
							max_relative_error,
							(max_relative_error < 1e-14 ? "OK" : "NG")
							);
}

int main() {
	std::vector<data_t> x_vec(N + 32), y_vec(N + 32), z_vec(N + 32);
	auto x_ptr = x_vec.data();
	auto y_ptr = y_vec.data();
	auto z_ptr = z_vec.data();
	while(reinterpret_cast<std::uint64_t>(x_ptr) % 32) {x_ptr++;}
	while(reinterpret_cast<std::uint64_t>(y_ptr) % 32) {y_ptr++;}
	while(reinterpret_cast<std::uint64_t>(z_ptr) % 32) {z_ptr++;}
#pragma omp parallel for
	for (std::size_t i = 0; i < N; i++) {
		x_ptr[i] = i + 1;
		y_ptr[i] = i + 1;
	}

	eval_core<data_t, 1   >(z_ptr, x_ptr, y_ptr, N);
	eval_core<data_t, 2   >(z_ptr, x_ptr, y_ptr, N);
	eval_core<data_t, 4   >(z_ptr, x_ptr, y_ptr, N);
	eval_core<data_t, 8   >(z_ptr, x_ptr, y_ptr, N);
	eval_core<data_t, 16  >(z_ptr, x_ptr, y_ptr, N);
	eval_core<data_t, 32  >(z_ptr, x_ptr, y_ptr, N);
	eval_core<data_t, 64  >(z_ptr, x_ptr, y_ptr, N);
	eval_core<data_t, 128 >(z_ptr, x_ptr, y_ptr, N);
	eval_core<data_t, 256 >(z_ptr, x_ptr, y_ptr, N);
	eval_core<data_t, 512 >(z_ptr, x_ptr, y_ptr, N);
	eval_core<data_t, 1024>(z_ptr, x_ptr, y_ptr, N);
}
