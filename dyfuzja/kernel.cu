
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <QtGui/QImage>
#include <QColor>
#include <iostream>

const double k = 1;
const double k2 = k * k;

cudaError_t addWithCuda(int *c, const int *a, const int *b, size_t size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

void dyfuzja(QImage &in, QImage &out, double delta);
//parametry:
// - s - wartosc
// - ss - kwadrat wartosci
double normaGradientu(double kwadratPochodnaX, double kwadratPochodnaY);
double pochodnaX(QImage &in, int x, int y);
double pochodnaY(QImage &in, int x, int y);
double pochodnaXX(QImage &in, int x, int y);
double pochodnaYY(QImage &in, int x, int y);

__global__ void dyfuzjaKernel(int *wyn, const int *data, const double delta, const int width)
{
	int x = threadIdx.x + 1;
	int y = threadIdx.y + 1;
	int prev = data[x + y * width];

	const double k = 1;
	const double k2 = k * k;
	
	double ux = ((double)(data[x + 1 + y * width] - data[x - 1 + y * width])) / 2;
	double uy = ((double)(data[x + (y + 1) * width] - data[x + (y - 1) * width])) / 2;
	double ux2 = ux * ux;
	double uy2 = uy * uy;
	double uxx = data[x + 1 + y * width] - 2 * data[x + y * width] + data[x + 1 + y * width];
	double uyy = data[x + (y + 1) * width] - 2 * data[x + y * width] + data[x + (y - 1) * width];
	double c = 1 / (1 + (ux2 + uy2) / k2);
	wyn[x + y * width] = (int)(prev + delta * ((c - c * c * 2 / k2) * (ux2 * uxx + uy2 * uyy)));
}

int main(int argc, char **argv)
{
	if(argc < 2)
	{
		std::cerr << "prosze podac sciezke do obrazka!\n";
		return 3;
	}
	for(unsigned u = 0; u < argc; u++)
	{
		std::cout << argv[u] << "\n";
	}

	std::cout<<"RUN!\n";
	QImage img;
	double dt = 0.01;
	if(!img.load(argv[1]))
	{
		std::cerr << "bledny plik!\n";
		return 1;
	}

	 QVector<QRgb> cTable1(256), cTable2(256);
	 for(unsigned u = 0; u < 256; u++)
	 {
		QColor color(u,u,u);
		cTable1[u] = color.rgb();
		cTable2[u] = color.rgb();
	 }
	
	QImage out(img.width(), img.height(), QImage::Format_Indexed8), 
		in(img.width(), img.height(), QImage::Format_Indexed8);
		
	out.setColorTable(cTable1);
	in.setColorTable(cTable2);
		
	out.fill(0);
	in.fill(0);
	
	std::cout<< "sprawdzam przestrzen barw.\n";
	
	if(!img.isGrayscale())
	{
		if(img.format() != QImage::Format_RGB32 && img.format() != QImage::Format_ARGB32)
		{
			std::cerr << "zly format pliku - oczekuje pliku w skali szarosci albo RGB 32 lub 24 bity, otrzymano: " << img.format() << " - sprawdz wartosc w QImage::Format\n";
			return 2;
		}
		double gray;
		unsigned tmp;
		QColor color;
		for(int x = 0; x < img.width(); x++)
		{
			for(int y = 0; y < img.height(); y++)
			{
				color = img.pixel(x, y);
				tmp = color.red();
				gray = tmp;
				tmp = color.green();
				gray += tmp;
				tmp = color.blue();
				gray += tmp;
				gray = gray / 3;
				
				in.setPixel(x, y, gray);
			}
		}
		
//		in = img.convertToFormat(QImage::Format_Indexed8);
	}
	else
		in = img;
		
	int *data = new int[img.width() * img.height()];
	for(unsigned x = 0; x < img.width(); x++)
	{
		for(unsigned y = 0; y < img.height(); y++)
		{
			QColor color(img.pixel(x, y));
			data[x + y * img.width()] = color.blue();
		}
	}

	std::cout<<"przystepuje do dyfuzji\n";
	
	unsigned min = 255, max = 0;
	for(unsigned x = 0; x < in.width(); x++)
	{
		for(unsigned y = 0; y < in.height(); y++)
		{
			QColor pix(in.pixel(x, y));
			if(pix.blue() < min)
				min = pix.blue();
			if(pix.blue() > max)
				max = pix.blue();
		}
	}
	
	std::cout << "max: " << max << "\tmin: " << min << "\nis gray scale: " << in.isGrayscale() << "\nis gray scale: " << out.isGrayscale() << "\n";
		
	in.save("C:/Users/Krzych/Desktop/gray.jpg");
	for(unsigned u = 0; u < 1; u++)
	{
		std::cout << "iteracja: " << u << "\n";
		dyfuzja(in, out, dt);
		in = out.copy();
	}
	
	std::cout << "skonczylem przetwazac\n";
	
	out.save("C:/Users/Krzych/Desktop/out_img.jpg");

	//--- to samo na CUDA
	//wskaŸniki dla zmiennych wykorzystywanych w kernelu
	int *dev_in = 0;
	int *dev_out = 0;
	int size = img.width() * img.height();
	
	//wybranie karty
	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	//alokowanie pamiêci karty graficznej
	cudaStatus = cudaMalloc((void**)&dev_in, size * sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_out, size * sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	
	//skopiowanie wartoœci do pamiêci karty graficznej
	cudaStatus = cudaMemcpy(dev_in, data, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	int wi = img.width();
	int he = img.height();
	
	//Uruchomienie kernela na karcie grafiki
	dim3 pixle(wi - 2, he - 2);
	dyfuzjaKernel<<<1, pixle>>>(dev_out, dev_in, dt, wi);

	//Synchronizacja z kart¹
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}
	
	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(data, dev_out, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	for(unsigned x = 0; x < wi; x++)
	{
		for(unsigned y = 0; y < he; y++)
		{
			out.setPixel(x, y, data[x + y * wi]);
		}
	}
	out.save("C:/Users/Krzych/Desktop/out_cuda_img.jpg");
	
Error:
	cudaFree(dev_in);
	cudaFree(dev_out);
	
	//czyszczenie CUDA
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
/*
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
*/
    return 0;
}

void dyfuzja(QImage &in, QImage &out, double delta)
{
	double nextVal;
	double c;
	double u, ux, ux2, uy, uy2, uxx, uyy;
	QColor color;
	for(int x = 0; x < in.width(); x++)
	{
		for(int y = 0; y < in.height(); y++)
		{
			color.setRgb(in.pixel(x, y));
			u = color.blue();
			ux = pochodnaX(in, x, y);
			uy = pochodnaY(in, x, y);
			ux2 = ux * ux;
			uy2 = uy * uy;
			uxx = pochodnaXX(in, x, y);
			uyy = pochodnaYY(in, x, y);
			c = normaGradientu(ux2, uy2);
			nextVal = u + delta * ((c - c * c * 2 / k2) * (ux2 * uxx + uy2 * uyy));
//			color.setRgb(nextVal, nextVal, nextVal);
//			std::cout << nextVal << "\n";
			out.setPixel(x, y, ((int)(nextVal))%256);
		}
	}
}

double normaGradientu(double kwadratPochodnaX, double kwadratPochodnaY)
{
	return 1 / (1 + (kwadratPochodnaX + kwadratPochodnaY) / k2);
}

double pochodnaX(QImage &in, int x, int y)
{
	QColor pix;
	if(x == 0 || y == 0 || x == in.width() - 1 || y == in.height() - 1)
	{
		pix.setRgb(in.pixel(x, y));
		return pix.blue();
	}
	
	QColor prev(in.pixel(x - 1, y)), next(in.pixel(x + 1, y));
	
	return ((double)(next.blue() - prev.blue())) / 2;
}

double pochodnaY(QImage &in, int x, int y)
{
	QColor pix(in.pixel(x, y));
	if(x == 0 || y == 0 || x == in.width() - 1 || y == in.height() - 1)
	{
		return pix.blue();
	}
	
	QColor prev(in.pixel(x, y - 1)), next(in.pixel(x, y + 1));
	
	return ((double)(next.blue() - prev.blue())) / 2;
}

double pochodnaXX(QImage &in, int x, int y)
{	
	QColor curr(in.pixel(x,y));
	if(x == 0 || y == 0 || x == in.width() - 1 || y == in.height() - 1)
		return curr.blue();
		
	QColor prev(in.pixel(x - 1, y)), next(in.pixel(x + 1, y));
	
	return next.blue() - 2 * curr.blue() + prev.blue();
}

double pochodnaYY(QImage &in, int x, int y)
{
	QColor curr(in.pixel(x,y));
	if(x == 0 || y == 0 || x == in.width() - 1 || y == in.height() - 1)
		return curr.blue();
		
	QColor prev(in.pixel(x, y - 1)), next(in.pixel(x, y + 1));
	
	return next.blue() - 2 * curr.blue() + prev.blue();
}


//------------------------------------------------------

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, size_t size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
