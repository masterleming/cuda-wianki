
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <QtGui/QImage>
#include <QColor>
#include <iostream>
#include <ctime>
#include <sstream>

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
void helpMe();
bool params();

// Zmienne globalne
int startX = -1, startY = -1, finishX = -1, finishY = -1;
unsigned cnt = -1;

__global__ void dyfuzjaKernel(unsigned *wyn, const unsigned *data, const int width, const int height, bool *czy)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	unsigned min = 0xffffffff;
	unsigned black = 0xff000000;
	
	if(data[x + y * width] != 0xff000000)
	{
		wyn[x + y * width] = data[x + y * width];
		return;
	}
	
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
	// int startX = -1, startY = -1, finishX = -1, finishY = -1;
	// unsigned cnt = -1;
	std::string imgDir = "";
	std::stringstream ss;
	
	for(int i = 0; i < argc; i++)
	{
		std::string s(argv[i]);
		if(s[0] != '-')
			continue;
			
		switch(s[1])
		{
			case 'i':
				if(s[2] != '=')
					continue;
				imgDir = s.substr(3);
				break;
			case 's':
				if(s[3] != '=')
					continue;
				if(s[2] == 'x')
				{
					ss.clear();
					ss.str(s.substr(4));
					ss >> startX;
				}
				else if(s[2] == 'y')
				{
					ss.clear();
					ss.str(s.substr(4));
					ss >> startY;
				}
				else
					continue;
				break;
			case 'f':
				if(s[3] != '=')
					continue;
				if(s[2] == 'x')
				{
					ss.clear();
					ss.str(s.substr(4));
					ss >> finishX;
				}
				else if(s[2] == 'y')
				{
					ss.clear();
					ss.str(s.substr(4));
					ss >> finishY;
				}
				else
					continue;
				break;
			case 'h':
				helpMe();
				return 0;
			case 'n':
				ss.clear();
				ss.str(s.substr(3));
				ss >> cnt;
		}
	}
	
	if(imgDir.size() == 0)
	{
		helpMe();
		return -1;
	}
	
	if(startX == -1 || startY == -1 || finishX == -1 || finishY == -1)
		while(params());
		
	if(startX == -1 || startY == -1 || finishX == -1 || finishY == -1)
	{
		helpMe();
		return 6;
	}

	// if(argc < 2)
	// {
		// std::cerr << "prosze podac sciezke do obrazka!\n";
		// return 3;
	// }
	// for(int u = 0; u < argc; u++)
	// {
		// std::cout << argv[u] << "\n";
	// }

	// std::cout<<"RUN!\n";
	QImage img;
	if(!img.load(imgDir.c_str()))
	{
		std::cerr << "bledny plik!\n";
		helpMe();
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
	
	//time_t timer1, timer2;
	
	// std::cout<<"przystepuje do dyfuzji\n";
	in.save("out/gray.png");
	/*	//to samo na CPU
	time(&timer1);
	dyfuzja(in, out);
	time(&timer2);
	
	double sec = difftime(timer2, timer1);
	
	out.save("out/out_dyf.png");
	narysujDroge(out, startX, startY, finishX, finishY);
	out.save("out/out_route.png");
	
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
	
	int wi = img.width();
	int he = img.height();
	
	if(startX >= wi || startY >= he || finishX >= wi || finishY >= he)
	{
		helpMe();
		return -2;
	}
	
	//Warunek musi byæ spe³niony, aby próbowaæ w ogóle rozwi¹zaæ zadanie
	if(wi % 32 != 0 || he % 32 != 0)
	{
		std::cerr << "obrazek ma wymiary nie bedace wielokrotnoscia 32 albo jest mniejszy niz 32x32!\n";
		helpMe();
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
	
	for(int i = 0; i < cnt; i++)
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
		
		// cudaStatus = cudaMemcpy(dev_in, dev_out, size * sizeof(unsigned), cudaMemcpyDeviceToDevice);
		// if (cudaStatus != cudaSuccess)
		// {
			// fprintf(stderr, "cudaMemcpy failed!");
			// goto Error;
		// }
		
		tmp_ptr = dev_in;
		dev_in = dev_out;
		dev_out = tmp_ptr;
		
		
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
	
	for(int x = 0; x < wi; x++)
	{
		for(int y = 0; y < he; y++)
		{
			// std::cout << data[x + y * wi] << "\t";
			out.setPixel(x, y, data[x + y * wi]);
		}
	}
	out.save("out/out_cuda_img.png");
	narysujDroge(out, startX, startY, finishX, finishY);
	out.save("out//out_cuda_img_rout.png");
	
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
	
	if(min == in.pixel(currX, currY))
		ret = DIR::none;
	
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

void helpMe()
{
	std::cout << "\nProgram poddaje obraz binarny dyfuzji w celu znalezienia najkrotszej drogi z podanych wspolrzednych poczatkowych do koncowych." << "\n\nSposob uzycia:\n" << "\tdyfuzja [-parametr=wartosc] -i=sciezka/do.pliku" << "\n\nLista parametrow:\n" << "\t-i=sciezka/do.pliku\tobraz, ktory ma zostac poddany dyfuzji.\n" << "\t-sx=wartosc\t\twpsolzedne x punktu poczatkowego\n" << "\t-sy=wartosc\t\twpsolzedne y punktu poczatkowego\n" << "\t-fx=wartosc\t\twpsolzedne x punktu koncowego\n" << "\t-fy=wartosc\t\twpsolzedne y punktu koncowego\n" << "\t-n=wartosc\t\tilosc powtorzen algorytmu\n" << "\t-h\t\t\tekran pomocy.\n\n";
}

bool params()
{
	std::cout << "\nNIEKTORE PARAMETRY WEJSCIOWE NIE ZOSTALY ZAINICJOWANE!" <<
		"\nstartX = " << startX << "\tstartY = " << startY << "\tfinishX = " << finishX << "\tfinishY = " << finishY <<
		"\n\nCzy chesz teraz je uzupelnic?\n(Jesli wartosci nie zostana uzupelnione, to program zakonczy wykonanie)\n\n";

	std::cout << "\t(1) startX\n\t(2) startY\n\t(3) finishX\n\t(4) finishY\n\n\t(0) wyjdz\n\n";
	int x;
	std::cin >> x;
	std::cin.clear();
	switch(x)
	{
		case 1:
			std::cout << "\nPodaj nowa wartosc dla startX: ";
			std::cin >> startX;
			break;
		case 2:
			std::cout << "\nPodaj nowa wartosc dla startY: ";
			std::cin >> startY;
			break;
		case 3:
			std::cout << "\nPodaj nowa wartosc dla finishX: ";
			std::cin >> finishX;
			break;
		case 4:
			std::cout << "\nPodaj nowa wartosc dla finishY: ";
			std::cin >> finishY;
			break;
		case 0:
			return false;
	}
	return true;
}



// $(QTDIR)\include
// $(QTDIR)\include\QtCore
// $(QTDIR)\include\QtGui