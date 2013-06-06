
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <QtGui/QImage>
#include <QColor>
#include <iostream>
#include <ctime>

enum DIR
{
	none = -1,
	up = 0,
	right,
	down,
	left
};

// DIR znajdzMinSasiad(QImage &in, int currX, int currY);

void dyfuzja(QImage &in, QImage &out);
void narysujDroge(QImage &img, int startX, int startY, int finishX, int finishY);


__global__ void dyfuzjaKernel(unsigned *wyn, const unsigned *data, const int width, const int height, bool *czy)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	unsigned min = 0xffffffff;
	unsigned black = 0xff000000;
	
	if(data[x + y * width] != 0xff000000)
		return;
	
	if(x > 0)
		if(data[x - 1 + y * width] != black && data[x - 1 + y * width] < min)
			min = data[x - 1 + y * width];
			
	if(x < width -1)
		if(data[x + 1 + y * width] != black && data[x + 1 + y * width] < min)
			min = data[x + 1 + y * width];
	
	if(y > 0)
		if(data[x + (y - 1) * width] != black && data[x + (y - 1) * width] < min)
			min = data[x + (y - 1) * width];
	
	if(y < height - 1)
		if(data[x + (y + 1) * width] != black && data[x + (y + 1) * width] < min)
			min = data[x + (y + 1) * width];
	
	if(min != 0xffffffff)
	{
		min += 1;
		wyn[x + y * width] = min;
		// wyn[x + y * width] = 64;
		*czy = true;
	}
	else
		wyn[x + y * width] = 0xff000000;//data[x + y * width];
}

int main(int argc, char **argv)
{
	int startX = 4, startY = 4, finishX = 60, finishY = 60;

	if(argc < 2)
	{
		std::cerr << "prosze podac sciezke do obrazka!\n";
		return 3;
	}
	// for(int u = 0; u < argc; u++)
	// {
		// std::cout << argv[u] << "\n";
	// }

	std::cout<<"RUN!\n";
	QImage img;
	if(!img.load(argv[1]))
	{
		std::cerr << "bledny plik!\n";
		return 1;
	}

	 // QVector<QRgb> cTable1(256), cTable2(256);
	 // for(unsigned u = 0; u < 256; u++)
	 // {
		// QColor color(u,u,u);
		// cTable1[u] = color.rgb();
		// cTable2[u] = color.rgb();
	 // }
	
	QImage out(img.width(), img.height(), QImage::Format_RGB32), 
		in(img.width(), img.height(), QImage::Format_RGB32);
		
	// out.setColorTable(cTable1);
	// in.setColorTable(cTable2);
		
	out.fill(0);
	in.fill(0);
	
	// std::cout<< "sprawdzam przestrzen barw.\n";
	
	if(!img.isGrayscale())
	{
		// if(img.format() != QImage::Format_RGB32 && img.format() != QImage::Format_ARGB32)
		// {
			// std::cerr << "zly format pliku - oczekuje pliku w skali szarosci albo RGB 32 lub 24 bity, otrzymano: " << img.format() << " - sprawdz wartosc w QImage::Format\n";
			// return 2;
		// }
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
				color.setRgb(gray, gray, gray);
				in.setPixel(x, y, color.rgb());
			}
		}
	}
	else
		in = img;
	
	in.setPixel(startX, startY, 0xff000001);	
	
	unsigned *data = new unsigned[img.width() * img.height()];
	for(int x = 0; x < img.width(); x++)
	{
		for(int y = 0; y < img.height(); y++)
		{
			QColor color(in.pixel(x, y));
			data[x + y * img.width()] = color.rgb();
			// std::cout << data[x + y * img.width()] << "\t";
		}
	}
	
	time_t timer1, timer2;
	
	// std::cout<<"przystepuje do dyfuzji\n";
	in.save("C:/Users/Krzych/Desktop/gray.png");
	/*	//to samo na CPU
	time(&timer1);
	dyfuzja(in, out);
	time(&timer2);
	
	double sec = difftime(timer2, timer1);
	
	out.save("C:/Users/Krzych/Desktop/out_dyf.png");
	narysujDroge(out, startX, startY, finishX, finishY);
	out.save("C:/Users/Krzych/Desktop/out_route.png");
	
	// std::cout << "skonczylem przetwazac\n";

	std::cout << "CPU time: " << sec << " [s]\n";
	//*/
	
	//--- to samo na CUDA
	//wskaŸniki dla zmiennych wykorzystywanych w kernelu
	unsigned *dev_in = NULL;
	unsigned *dev_out = NULL;
	unsigned *tmp_ptr = NULL;
	bool *dev_stop_condition = NULL;
	unsigned size = img.width() * img.height();
	
	unsigned wi = img.width();
	unsigned he = img.height();
	
	//Warunek musi byæ spe³niony, aby próbowaæ w ogóle rozwi¹zaæ zadanie
	if(wi % 32 != 0 || he % 32 != 0)
	{
		std::cerr << "obrazek ma wymiary nie bedace wielokrotnoscia 32 albo jest mniejszy niz 32x32!\n";
		return 5;
	}
	
	//wybranie karty
	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	//alokowanie pamiêci karty graficznej
	cudaStatus = cudaMalloc((void**)&dev_in, size * sizeof(unsigned));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_out, size * sizeof(unsigned));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	
	cudaStatus = cudaMalloc((void**)&dev_stop_condition, sizeof(bool));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	
	//skopiowanie wartoœci do pamiêci karty graficznej
	cudaStatus = cudaMemcpy(dev_in, data, size * sizeof(unsigned), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	
	cudaStatus = cudaMemcpy(dev_out, dev_in, size * sizeof(unsigned), cudaMemcpyDeviceToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	
	bool *b_tmp = new bool;
	*b_tmp = false;
	cudaStatus = cudaMemcpy(dev_stop_condition, b_tmp, sizeof(bool), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	
	//Uruchomienie kernela na karcie grafiki
	dim3 threadsPerBlock(32, 32);	//Maximum supported values!
	dim3 numBlocks(wi / threadsPerBlock.x, he / threadsPerBlock.y);
	
	cudaEvent_t start, stop;
	float time = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	
	while(true)
	{
		dyfuzjaKernel<<<numBlocks, threadsPerBlock>>>(dev_out, dev_in, wi, he, dev_stop_condition);
		
		cudaStatus = cudaMemcpy(b_tmp, dev_stop_condition, sizeof(bool), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}
		
		if(*b_tmp == false)
			break;
		
		cudaStatus = cudaMemcpy(dev_in, dev_out, size * sizeof(unsigned), cudaMemcpyDeviceToDevice);
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}
		
		*b_tmp = false;
		cudaStatus = cudaMemcpy(dev_stop_condition, b_tmp, sizeof(bool), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}
	}
	
	//Synchronizacja z kart¹
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	
	std::cout << "GPU time: " << time << " [ms]\n";
	
	for(unsigned u = 0; u < size; u++)
		data[u] = 0;
	
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
	out.save("C:/Users/Krzych/Desktop/out_cuda_img.png");
	narysujDroge(out, startX, startY, finishX, finishY);
	out.save("C:/Users/Krzych/Desktop/out_cuda_img_rout.png");
	
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

unsigned znajdzMin(QImage &in, int currX, int currY)
{
	unsigned min = 0xfffffffe;
	unsigned color;
	QColor black(0, 0, 0);
	if(currX > 0)
	{
		color = in.pixel(currX - 1, currY);
		if(color != black.rgb())
			if(min > color)
				min = color;
	}
	
	if(currX < in.width() - 1)
	{
		color = in.pixel(currX + 1, currY);
		if(color != black.rgb())
			if(min > color)
				min = color;
	}

	if(currY > 0)
	{
		color = in.pixel(currX, currY - 1);
		if(color != black.rgb())
			if(min > color)
				min = color;
	}

	if(currY < in.height() - 1)
	{
		color = in.pixel(currX, currY + 1);
		if(color != black.rgb())
			if(min > color)
				min = color;
	}
	return min + 1;
}

DIR znajdzMinSasiad(QImage &in, int currX, int currY)
{
	DIR ret = DIR::none;
	unsigned min = 0xffffffff;
	unsigned color;
	QColor black(0, 0, 0);
	if(currX > 0)
	{
		color = in.pixel(currX - 1, currY);
		if(min > color)
		{
			min = color;
			ret = DIR::left;
		}
	}
	
	if(currX < in.width() - 1)
	{
		color = in.pixel(currX + 1, currY);
		if(min > color)
		{
			min = color;
			ret = DIR::right;
		}
	}

	if(currY > 0)
	{
		color = in.pixel(currX, currY - 1);
		if(min > color)
		{
			min = color;
			ret = DIR::up;
		}
	}

	if(currY < in.height() - 1)
	{
		color = in.pixel(currX, currY + 1);
		if(min > color)
		{
			min = color;
			ret = DIR::down;
		}
	}
	return ret;
}

void dyfuzja(QImage &in, QImage &out)
{
	// in.setPixel(startX, startY, 0xff000001);
	out = in;

	QColor black(0, 0, 0);
	// std::cout << "black: " << std::hex << black.rgb() << "\n";
	bool czy = false;
	while(true)
	{
		for(int x = 0; x < in.width(); x++)
		{
			for(int y = 0; y < in.height(); y++)
			{
				unsigned c = out.pixel(x, y);
				if(c == black.rgb())
				{
					unsigned min = znajdzMin(in, x, y);
					if(min != 0xffffffff)
					{
						c = min;
						out.setPixel(x, y, c);
						czy = true;
					}
				}
			}
		}
		
		if(!czy)
			break;
		czy = false;
		in = out;
	}
}

void narysujDroge(QImage &img, int startX, int startY, int finishX, int finishY)
{
	QColor green(0, 255, 0);
	QColor blue(0, 0, 255);
	QColor red(255, 0, 0);
	QColor mazak(0xff, 80, 80);
	int x = finishX, y = finishY;
	QImage tmp = img;

	while(x != startX || y != startY)
	{
		DIR next = znajdzMinSasiad(tmp, x, y);
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
		img.setPixel(x, y, mazak.rgb()); 
	}
	img.setPixel(startX, startY, green.rgb());
	img.setPixel(finishX, finishY, red.rgb());
}
