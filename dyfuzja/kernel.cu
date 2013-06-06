
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <QtGui/QImage>
#include <QColor>
#include <iostream>

enum DIR
{
	none = -1,
	up = 0,
	right,
	down,
	left
};

// DIR znajdzMinSasiad(QImage &in, int currX, int currY);

void dyfuzja(QImage &in, QImage &out, int startX, int startY, int finishX, int finishY);


__global__ void dyfuzjaKernel(int *wyn, const int *data, const double delta, const int width)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	wyn[x + y * width] = 16 * (blockIdx.x + blockIdx.y);
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
	float dt = 0.01;
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
	}
	else
		in = img;
		
	int *data = new int[img.width() * img.height()];
	for(unsigned x = 0; x < img.width(); x++)
	{
		for(unsigned y = 0; y < img.height(); y++)
		{
			QColor color(in.pixel(x, y));
			data[x + y * img.width()] = color.blue();
			// std::cout << data[x + y * img.width()] << "\t";
		}
	}

	std::cout<<"przystepuje do dyfuzji\n";
	in.save("C:/Users/Krzych/Desktop/gray.png");
	
	dyfuzja(in, out, 4, 4, 60, 60);
	std::cout << "skonczylem przetwazac\n";
	
	out.save("C:/Users/Krzych/Desktop/out_img.png");

	return 0;
	
	//--- to samo na CUDA
	//wskaŸniki dla zmiennych wykorzystywanych w kernelu
	int *dev_in = NULL;
	int *dev_out = NULL;
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
	
	if(wi % 32 != 0 || he % 32 != 0)
	{
		std::cerr << "obrazek ma wymiary nie bedace wielokrotnoscia 32 albo sa mniejsze!\n";
		return 5;
	}
	
	//Uruchomienie kernela na karcie grafiki
	dim3 threadsPerBlock(32, 32);	//Maximum supported values!
	dim3 numBlocks(wi / threadsPerBlock.x, he / threadsPerBlock.y);
	dyfuzjaKernel<<<numBlocks, threadsPerBlock>>>(dev_out, dev_in, dt, wi);

	//Synchronizacja z kart¹
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}
	
	for(unsigned u = 0; u < size; u++)
		data[u] = 200;
	
	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(data, dev_out, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	
	out.fill(0);
	if(!out.isGrayscale())
	{
		std::cerr << "out nie jest szaroodcieniowe!";
		return 4;
	}
	
	for(unsigned x = 0; x < wi; x++)
	{
		for(unsigned y = 0; y < he; y++)
		{
			// std::cout << data[x + y * wi] << "\t";
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

int znajdzMin(QImage &in, int currX, int currY)
{
	int min = 255;
	QColor color;
	if(currX > 0)
	{
		color = in.pixel(currX - 1, currY);
		if(color.blue() != 0)
			if(min > color.blue())
				min = color.blue();
	}
	
	if(currX < in.width() - 1)
	{
		color = in.pixel(currX + 1, currY);
		if(color.blue() != 0)
			if(min > color.blue())
				min = color.blue();
	}

	if(currY > 0)
	{
		color = in.pixel(currX, currY - 1);
		if(color.blue() != 0)
			if(min > color.blue())
				min = color.blue();
	}

	if(currY < in.height() - 1)
	{
		color = in.pixel(currX, currY + 1);
		if(color.blue() != 0)
			if(min > color.blue())
				min = color.blue();
	}
	return min + 1;
}

DIR znajdzMinSasiad(QImage &in, int currX, int currY)
{
	DIR ret = DIR::none;
	int min = 255;
	QColor color;
	if(currX > 0)
	{
		color = in.pixel(currX - 1, currY);
		// if(color.blue() != 0)
			if(min > color.blue())
			{
				min = color.blue();
				ret = DIR::left;
			}
	}
	
	if(currX < in.width() - 1)
	{
		color = in.pixel(currX + 1, currY);
		// if(color.blue() != 0)
			if(min > color.blue())
			{
				min = color.blue();
				ret = DIR::right;
			}
	}

	if(currY > 0)
	{
		color = in.pixel(currX, currY - 1);
		// if(color.blue() != 0)
			if(min > color.blue())
			{
				min = color.blue();
				ret = DIR::up;
			}
	}

	if(currY < in.height() - 1)
	{
		color = in.pixel(currX, currY + 1);
		// if(color.blue() != 0)
			if(min > color.blue())
			{
				min = color.blue();
				ret = DIR::down;
			}
	}
	return ret;
}

void dyfuzja(QImage &in, QImage &out, int startX, int startY, int finishX, int finishY)
{
	in.setPixel(startX, startY, 1);
	out = in;

	bool czy = false;
	while(true)
	{
		for(unsigned x = 0; x < in.width(); x++)
		{
			for(unsigned y = 0; y < in.height(); y++)
			{
				QColor c = out.pixel(x, y);
				if(c.blue() == 0)
				{
					int min = znajdzMin(in, x, y);
					c.setRgb(min, min, min);
					out.setPixel(x, y, c.rgb());
					czy  = true;
				}
			}
		}
		
		if(!czy)
			break;
		czy = false;
		in = out;
	}
	
	out.convertToFormat(QImage::Format_RGB32);
	QColor c(0, 255, 0);
	out.setPixel(startX, startY, c.rgb());
	c.setRgb(255, 0, 255);
	out.setPixel(finishX, finishY, c.rgb());
	
	c.setRgb(out.pixel(finishX, finishY));
	QColor green(0, 255, 0);
	QColor blue(0, 0, 255);
	int x = finishX, y = finishY;
	while(c != green)
	{
		DIR next = znajdzMinSasiad(out, x, y);
		switch(next)
		{
			case DIR::up:
				y -= 1;
				break;
			case DIR::right:
				x += 1;
				break;
			case DIR::down:
				y += 1;
				break;
			case DIR::left:
				x -= 1;
				break;
			case DIR::none:
				std::cerr << "nieznany kierunek!\n";
				return;
		}
		c.setRgb(out.pixel(x, y));
		out.setPixel(x, y, blue.rgb()); 
	}
}
