#include <opencv2\opencv.hpp>
#include <iomanip>
#include <iostream>
#include <string.h>
#include <time.h>
#include <cstring>
#include <cstdio>
#define SN 2             //SSIM窗口大小为2*SN+1

using namespace std;
using namespace cv;

template<class T>class Image  //C++像素访问外壳
{
private:
	IplImage* imgp;
public:
	Image(IplImage* img=0){imgp=img;}
	~Image(){imgp=0;}
	void operator=(IplImage* img){imgp=img}
	inline T* operator[](const int rowlndx)
	{
		return((T*)(imgp->imageData+rowlndx*imgp->widthStep));
	}
};

template <typename T>
T abs(T a,T b)
{
	return ((a-b)>0)?(a-b):(b-a);
}

double TR=0.067;
typedef Image<float> BwImageFloat;
typedef Image<unsigned char> BwImage;
typedef struct
{
	unsigned char b,g,r;
}RgbPixel;

typedef struct
{
	float b,g,r;
}RgbPixelFloat;

typedef Image<RgbPixel>  RgbImage;
typedef Image<RgbPixelFloat>  RgbImageFloat;

double floatimage_mean(IplImage* &img);
double image_mean(IplImage* &img);
double image_mean(IplImage* img,IplImage* ROI);
IplImage* ssim(IplImage* x,IplImage* y,double sigma=5.5);
IplImage* dev(IplImage* src,IplImage* avg,double sigma=0.5);
IplImage* cov(IplImage* srcref,IplImage* avgref,IplImage* srcdis,IplImage* avgdis,double sigma=1.2);

template<typename T>
bool addchar(T cp,char* name,char* cc);


void main()
{

	//CvCapture* capture = cvCreateFileCapture( "H:\\avi\\1.avi");
	IplImage* frame;
	char temp[20] = {0};
	char outputFileName[512] = {0};

		//strcpy(outputFileName, outputFileNamePre);
		//sprintf(temp, "%d.jpeg", ++i);
		//strcat(outputFileName, temp);

		//cvSaveImage( outputFileName, frame);  
		


//	while(1)  
//	{  
//		frame = cvQueryFrame(capture);  
//		if(!frame)
//			break ;
		//strcpy(outputFileName, frame);
		//strcat(outputFileName, frame);
		char imageNamePre[] = "F:\\vs\\样本\\正样本\\";
		char imageName[100] ;
	//	char *imageName = "E:\\VS2010 OpenCV3\\myproject\\Image\\";
	//	int i=0;
		char *fileName = "E:\\项目程序\\程序\\tiqu\\tezheng.txt";
	//	char *fileNameROI = "H:\\My project\\a_QualityAssessment\\Data\\featureValues\\chenweihuaROIFeatures.txt";
		char tempName[20];
		IplImage *cimg = NULL;
		IplImage *gimg = NULL;
	//	double start, finish;

		FILE *fp;
		FILE *fpROI;

		if( (fp = fopen(fileName,"a")) == NULL )
		{
			printf("open file error(1)\n");
			exit(1);
		}

		for(int i=1;i<=150;i++)
		{
			//start = clock();
			//int i = 235;
			strcpy(imageName,imageNamePre);
			sprintf(tempName,"%d.jpg",i);
			strcat(imageName,tempName);
		
			cimg = cvLoadImage(imageName, CV_LOAD_IMAGE_ANYDEPTH|CV_LOAD_IMAGE_ANYCOLOR); //无损载入图像
			if( cimg == NULL )
			{
				cout << "open file error(3)\n";
				exit(0);
			}

			int W = cimg->width, H = cimg->height;
			gimg = cvCreateImage(cvSize(W,H),IPL_DEPTH_8U,1);

			IplImage* vgimg=cvCreateImage(cvSize(W,H),IPL_DEPTH_32F,1);
			IplImage* hgimg=cvCreateImage(cvSize(W,H),IPL_DEPTH_32F,1);
			IplImage* simg=cvCreateImage(cvSize(W,H),IPL_DEPTH_32F,1);
			IplImage* EROI=cvCreateImage(cvSize(W,H),IPL_DEPTH_8U,1);//边缘特征（边缘）感兴趣区域
			IplImage* GROI=cvCreateImage(cvSize(W,H),IPL_DEPTH_8U,1);//梯度特征和熵特征感兴趣区域
			IplImage* egimg=cvCreateImage(cvSize(W,H),IPL_DEPTH_8U,1);

			cvCvtColor(cimg,gimg,CV_BGR2GRAY);	//cing->ging  彩色图像转灰色图像

			BwImage pgimg(gimg);



			//////////开始提取特征值////////////

			////感兴趣区域提取
			cvScale(gimg,simg);
			for(int i=0;i<12;++i)
				cvSmooth( simg, simg,CV_BLUR, 5,5);

			cvSobel(simg, hgimg, 1, 0, 3 );
			cvSobel(simg, vgimg, 0, 1, 3 );

			cvPow(hgimg,hgimg,2);
			cvPow(vgimg,vgimg,2);
			cvAdd(hgimg,vgimg,simg);
			cvPow(simg,simg,0.5);

			double avg=floatimage_mean(simg);//avg表示平滑后梯度图像的均值

			///二值化
			cvThreshold(simg, EROI, avg*TR,255,CV_THRESH_BINARY);
			cvAdaptiveThreshold(gimg,GROI,255,CV_ADAPTIVE_THRESH_GAUSSIAN_C,CV_THRESH_BINARY_INV);

			//////////闭运算	
			const int ME=9,MG=5;//为开运算边长////////////////////////////////////////最好是奇数，这样锚点可以选在选在其中央
			IplConvKernel* elemE=cvCreateStructuringElementEx( ME,ME,ME/2,ME/2,CV_SHAPE_ELLIPSE);
			IplConvKernel* elemG=cvCreateStructuringElementEx( MG,MG,MG/2,MG/2,CV_SHAPE_ELLIPSE);

			cvDilate(EROI,EROI,elemE,1);////膨胀（白扩散）
			cvErode(EROI,EROI,elemE,1);////腐蚀（黑扩散）

			cvDilate(GROI,GROI,elemG,1);
			cvErode(GROI,GROI,elemG,1);

			double MEROI=image_mean(EROI);
			double MGROI=image_mean(GROI);


			double eroi=MEROI/255;//感兴趣区域面积比
			double groi=MGROI/255;//测试用，groi总要比eroi分割的严格，即groi<eroi



				////////////////特征提取
				////////边缘特征提取
				cvCanny( gimg, egimg, 0,0,3);//双阈值在此设定,最后一个参数为Sobel算子的内核大小。第三第四个参数为阈值，函数能自动将第三第四个参数中大的那个作为高阈值 小的那个作为低阈值
				BwImage pegimg(egimg);
			//	if(flag_wether_showtexture==true)
			//	{
			//		char imagename[256];
			//		addchar(fname,"(edge)",imagename);
			//		show("edge",egimg);
			//	}
				double eavg=0,_eavg=0;
				eavg=image_mean(egimg);//未分割边缘特征1:eavg
				_eavg=image_mean(egimg,EROI);//分割边缘特征1:_eavg

			//	cout << setw(15) << left << "edge VF:" << _eavg << endl;
				fprintf(fp,"0 1:%010.6f\t",eavg);
			//	fprintf(fpROI,"%010.6f\t",_eavg);

				////////熵特征提取
				BwImage pGROI(GROI);
				BwImage pEROI(EROI);
				double histogram[256]={0},_histogram[256]={0};
				long sq=gimg->height*gimg->width,count=0;
				if(eroi<0.1)//如果eroi过小（<10%），说明图像非常模糊或者没有明确内容，此时不进行分割。
				{
					for(int i=0;i<H;++i)
						for(int j=0;j<W;++j)
						{
							int t=pgimg[i][j];
							histogram[t]++;
							_histogram[t]++;
							++count;
						}
				}
				else
				{
					for(int i=0;i<H;++i)
						for(int j=0;j<W;++j)
						{
							int t=pgimg[i][j];
							histogram[t]++;
							if(pGROI[i][j]!=0)
							{
								int t=pgimg[i][j];
								_histogram[t]++;
								++count;
							}
						}
				}

				double entropy=0,_entropy=0;
				for(int k=0;k<256;++k)
				{
					if(histogram[k]!=0) entropy+=histogram[k]*logf(sq/(histogram[k]))/logf(2)/sq;//未分割熵 特征2:entropy	
					if(_histogram[k]!=0) _entropy+=_histogram[k]*logf(count/(_histogram[k]))/logf(2)/count;//分割熵 特征2:_entropy		
				}

			//	cout <<  setw(15) << left << "entropy VF: " << entropy << endl;
				fprintf(fp,"2:%010.6f\t",entropy);
			//	fprintf(fpROI,"%010.6f\t",_entropy);

				////////梯度特征提取
				cvSobel(gimg, hgimg, 1, 0, 3 );
				cvSobel(gimg, vgimg, 0, 1, 3 );	

				cvPow(hgimg,hgimg,2);
				cvPow(vgimg,vgimg,2);
				cvAdd(hgimg,vgimg,simg);
				cvPow(simg,simg,0.5);

			//	if(flag_wether_showgradient==true)
			//	{
			//		char imagename[256];
			//		addchar(fname,"(gradient)",imagename);
			//		show("gradient",simg);
			//	}
				double gavg=0,_gavg=0;
				gavg=image_mean(simg);//未分割特征3:gavg/////////////////////////////////////////////
				_gavg=image_mean(simg,EROI);//分割特征3:_gavg////////////////////////////////////////////////////

			//	cout << setw(15) << left << "gradient VF: " << _gavg << endl;
				fprintf(fp,"3:%010.6f\t",gavg);
			//	fprintf(fpROI,"%010.6f\t",_gavg);
				//////////SSIM特征提取
				IplImage* sub1=cvCreateImage(cvSize(W/2,H/2),IPL_DEPTH_8U,1);
				IplImage* sub2=cvCreateImage(cvSize(W/2,H/2),IPL_DEPTH_8U,1);
				IplImage* sub3=cvCreateImage(cvSize(W/2,H/2),IPL_DEPTH_8U,1);
				IplImage* sub4=cvCreateImage(cvSize(W/2,H/2),IPL_DEPTH_8U,1);

				BwImage p1(sub1);
				BwImage p2(sub2);
				BwImage p3(sub3);
				BwImage p4(sub4);

				////下1/2采样
				for(int i=0;i<H/2*2;++i)
					for(int j=0;j<W/2*2;++j)
					{
						if(i%2==0)
						{
							if(j%2==0)	p1[i/2][j/2]=pgimg[i][j];
							else		p2[i/2][j/2]=pgimg[i][j];
						}
						else
						{
							if(j%2==0)	p3[i/2][j/2]=pgimg[i][j];
							else		p4[i/2][j/2]=pgimg[i][j];
						}
					}

					IplImage* subh=ssim(sub1,sub2);
					IplImage* subv=ssim(sub1,sub3);
					IplImage* subhv=ssim(sub1,sub4);
					IplImage* subvh=ssim(sub2,sub3);

				//	if(wether_show_subSSIM==true)
				//	{
				//		show("水平逆时针0度方向上的结构相似度图像",	subh	,0,1,"try");
				//		show("水平逆时针45度方向上的结构相似度图像",	subvh	,0,1,"try");
				//		show("水平逆时针90度方向上的结构相似度图像",	subv	,0,1,"try");
				//		show("水平逆时针135度方向上的结构相似度图像",	subhv	,0,1,"try");
				//	}
					double q[6];
					double u[5]={floatimage_mean(subh),floatimage_mean(subv),floatimage_mean(subhv),floatimage_mean(subvh)};
					sort(u,u+4);  //未进行平坦区域分割的各个相关性数值
					q[0]=(u[0]+u[1]+u[2]+u[3])/4;
					q[1]=(u[1]+u[2])/2;
					q[2]=u[3]-u[0];
					q[3]=(abs(u[0],q[0])+abs(u[1],q[0])+abs(u[2],q[0])+abs(u[3],q[0]))/4;
					q[4]=((u[0]-q[0])*(u[0]-q[0])+(u[1]-q[0])*(u[1]-q[0])+(u[2]-q[0])*(u[2]-q[0])+(u[3]-q[0])*(u[3]-q[0]))/4;
					q[5]=sqrt(q[4]);

					double _q[6];
					double _u[4]={floatimage_mean(subh),floatimage_mean(subv),floatimage_mean(subhv),floatimage_mean(subvh)};

					if(eroi>0.1)//如果eroi过小，表示图像一篇模糊，无法提取ROI，就不用近似计算分割了
					{
						for(int i=0;i<4;++i)
							_u[i]=(_u[i]+eroi-1)/eroi;
					}

					sort(_u,_u+4);	//平坦区域分割后的各个相关性数值
					_q[0]=(_u[0]+_u[1]+_u[2]+_u[3])/4;
					_q[1]=(_u[1]+_u[2])/2;
					_q[2]=_u[3]-_u[0];
					_q[3]=(abs(_u[0],_q[0])+abs(_u[1],_q[0])+abs(_u[2],_q[0])+abs(_u[3],_q[0]))/4;
					//MAD
					_q[4]=((_u[0]-_q[0])*(_u[0]-_q[0])+(_u[1]-_q[0])*(_u[1]-_q[0])+(_u[2]-_q[0])*(_u[2]-_q[0])+(_u[3]-_q[0])*(_u[3]-_q[0]))/4;
					_q[5]=sqrt(_q[4]);

		//			cout << "texture VF : " << endl;
		//			cout <<  setw(15) << left << "    Min: " << _u[0] << endl;
		//			cout <<  setw(15) << left << "    SMin: " << _u[1] << endl;
		//			cout <<  setw(15) << left << "    MAD:" << _q[4] << endl;
					fprintf(fp,"4:%010.6f\t5:%010.6f\t6:%010.6f\n",u[0],u[1],q[4]);
			//		fprintf(fpROI,"%010.6f\t%010.6f\n",_u[0],_u[1],_q[4]);
		//					char c = cvWaitKey(1);  
//
	//				if(c == 27)  
	//					break;  
	//			}  
		//		cvReleaseCapture(&capture); 

				cvReleaseImage(&egimg);
				cvReleaseImage(&EROI);
				cvReleaseImage(&GROI);
				cvReleaseImage(&simg);
				cvReleaseImage(&hgimg);
				cvReleaseImage(&vgimg);
				cvReleaseImage(&sub1);
				cvReleaseImage(&sub2);
				cvReleaseImage(&sub3);
				cvReleaseImage(&sub4);
				cvReleaseImage(&subh);
				cvReleaseImage(&subv);
				cvReleaseImage(&subhv);
				cvReleaseImage(&subvh);

	//			finish = clock();

				printf("The %dth is over\n",i);
	//			printf("time is %f\n\n",(finish-start)/CLOCKS_PER_SEC);

		}
			
		cvReleaseImage(&cimg);
		cvReleaseImage(&gimg);

		fclose(fp);

		printf("game over\n");
		//以示区分
	
}



double floatimage_mean(IplImage* &img)
{
	double avg=0;
	BwImageFloat pimg(img);
	double accum=0;
	for(int y=0;y<img->width;++y)
		for(int x=0;x<img->height;++x)
			accum+=pimg[x][y];
	double sq=img->width*img->height;
	avg=accum/sq;
	return avg;
}

double image_mean(IplImage* &img)
{
	double avg=0;
	BwImage pimg(img);
	double accum=0;
	for(int y=0;y<img->width;++y)
		for(int x=0;x<img->height;++x)
			accum+=pimg[x][y];
	double sq=img->width*img->height;
	avg=accum/sq;
	return avg;
}


template<typename T>
bool addchar(T cp,char* name,char* cc)
{
	for(int i=0;;++i)
	{
		if(*(cp+i)=='\0')
		{
			for(int j=0;j<256;++j)
			{
				*(cc+i+j)=*(name+j);
				if(*(name+j)=='\0') return true;
			}
			return false;
		}
		else *(cc+i)=*(cp+i);
	}
}


void show(const char* imagename,const IplImage* img,IplImage* ROI,double scale,const char* postfix)
{
	char name[256]={'\0'};
	if(postfix!='\0')
	{
		int i=0,j=0;
		for(;*(imagename+i)!='\0';++i)
			name[i]=*(imagename+i);

		for(;*(postfix+j)!='\0';++j)
			name[i+j]=*(postfix+j);

		name[i+j]='\0';
	}
	else
	{
		int i=0;
		for(;*(imagename+i)!='\0';++i)
			name[i]=*(imagename+i);

		name[i]='\0';
	}

	static int xd=0,yd=0,tc=0;
	const int num=23;//越大保留窗口数=num-5
	static vector<string> tmp(num);
	BwImage pROI(ROI);

	for(int i=num-1;i>0;--i)
		tmp[i]=tmp[i-1];
	tmp[0]=name;

	if(tmp[num-1]!="")	cvDestroyWindow(tmp[num-1].c_str());
	tc++;
	if(xd==960) 
	{
		xd=0;
		yd+=450;
	}
	if(yd==900) yd=0;
	////开始分通道分位深来显示图像
	if(img->nChannels==1)
	{
		if(img->depth==8)	
		{
			IplImage* buf=cvCreateImage(cvSize(img->width,img->height),IPL_DEPTH_8U,1);
			buf=cvCloneImage(img);
			if(ROI!=NULL)
			{
				BwImage pbuf(buf);
				for(int i=0;i<buf->height;++i)
					for(int j=0;j<buf->width;++j)
					{
						if(pROI[i][j]==0)
							pbuf[i][j]=0;
					}
			}
			cvShowImage(name,buf);
			cvReleaseImage(&buf);
		}
		if((img->depth==32)||(img->depth==64))
		{
			IplImage* buf=cvCreateImage(cvSize(img->width,img->height),IPL_DEPTH_32F,1);
			buf=cvCloneImage(img);
			BwImageFloat pbuf(buf);
			for(int i=0;i<img->height;++i)
				for(int j=0;j<img->width;++j)
				{
					if(ROI==NULL) pbuf[i][j]=pbuf[i][j]/scale;
					else
					{
						if(pROI[i][j]==0)	pbuf[i][j]=0;
						else				pbuf[i][j]=pbuf[i][j]/scale;
					}
				}
				cvShowImage(name,buf);
				cvReleaseImage(&buf);
		}
	}
	else if(img->nChannels==3)
	{
		if(img->depth==8)	
		{
			IplImage* buf=cvCreateImage(cvSize(img->width,img->height),IPL_DEPTH_8U,3);
			buf=cvCloneImage(img);
			if(ROI!=NULL)
			{
				RgbImage pbuf(buf);
				for(int i=0;i<buf->height;++i)
					for(int j=0;j<buf->width;++j)
					{
						if(pROI[i][j]==0)
						{
							pbuf[i][j].r=0;
							pbuf[i][j].g=0;
							pbuf[i][j].b=0;
						}
					}
			}
			cvShowImage(name,buf);
			cvReleaseImage(&buf);
		}
		if((img->depth==32)||(img->depth==64))
		{
			IplImage* buf=cvCreateImage(cvSize(img->width,img->height),IPL_DEPTH_32F,3);
			buf=cvCloneImage(img);
			RgbImageFloat pbuf(buf);
			for(int i=0;i<img->height;++i)
				for(int j=0;j<img->width;++j)
				{
					if(ROI==NULL)
					{
						pbuf[i][j].r=pbuf[i][j].r/scale;
						pbuf[i][j].g=pbuf[i][j].g/scale;
						pbuf[i][j].b=pbuf[i][j].b/scale;
					}
					else
					{
						if(pROI[i][j]==0)
						{
							pbuf[i][j].r=0;
							pbuf[i][j].g=0;
							pbuf[i][j].b=0;
						}
						else
							pbuf[i][j].r=pbuf[i][j].r/scale;
						pbuf[i][j].g=pbuf[i][j].g/scale;
						pbuf[i][j].b=pbuf[i][j].b/scale;
					}
				}
				cvShowImage(name,buf);
				cvReleaseImage(&buf);
		}
	}

	cvMoveWindow(name,xd,yd);
	xd+=480;
}


double image_mean(IplImage* img,IplImage* ROI)
{
	if(image_mean(ROI)<2) return image_mean(img);
	else
	{
		double avg=0;
		double accum=0,acount=0;
		if(ROI->depth!=8)
		{
			cout<<"ROI位深必须得是8"<<endl;
			return -1;
		}
		BwImage pROI(ROI);
		BwImage pimg(img);
		for(int y=0;y<img->width;++y)
			for(int x=0;x<img->height;++x)
			{
				if(pROI[x][y]!=0)
				{
					accum+=pimg[x][y];
					++acount;
				}
			}
			avg=accum/acount;
			return avg;
	}
}

IplImage* ssim(IplImage* x,IplImage* y,double sigma)
{
	const float C1 = 6.5025, C2 = 58.5225;
	IplImage* imgx=cvCreateImage(cvSize(x->width,x->height),IPL_DEPTH_32F,1);
	cvScale(x,imgx);
	IplImage* imgy=cvCreateImage(cvSize(x->width,x->height),IPL_DEPTH_32F,1);
	cvScale(y,imgy);

	IplImage* ux=cvCreateImage(cvSize(imgx->width,imgx->height),IPL_DEPTH_32F,1);
	cvSmooth(imgx,ux,CV_BLUR,2*SN+1,2*SN+1);
	BwImageFloat pux(ux);
	IplImage* uy=cvCreateImage(cvSize(imgx->width,imgx->height),IPL_DEPTH_32F,1);
	cvSmooth(imgy,uy,CV_BLUR,2*SN+1,2*SN+1);
	BwImageFloat puy(uy);

	IplImage* devx=dev(imgx,ux,sigma);
	BwImageFloat pdevx(devx);
	IplImage* devy=dev(imgy,uy,sigma);
	BwImageFloat pdevy(devy);

	IplImage* covxy=cov(imgx,ux,imgy,uy,sigma);
	BwImageFloat pcov(covxy);

	IplImage* SSIM=cvCreateImage(cvSize(imgx->width-2*SN,imgx->height-2*SN),IPL_DEPTH_32F,1);
	BwImageFloat pssim(SSIM);
	for(int i=0;i<SSIM->height;i++)
		for(int j=0;j<SSIM->width;j++)
			pssim[i][j]=(2*pux[i+SN][j+SN]*puy[i+SN][j+SN]+C1)*(2*pcov[i+SN][j+SN]+C2)/
			(pux[i+SN][j+SN]*pux[i+SN][j+SN]+puy[i+SN][j+SN]*puy[i+SN][j+SN]+C1)/(pdevx[i+SN][j+SN]*pdevx[i+SN][j+SN]+pdevy[i+SN][j+SN]*pdevy[i+SN][j+SN]+C2);

	cvReleaseImage(&imgx);
	cvReleaseImage(&imgy);
	cvReleaseImage(&ux);
	cvReleaseImage(&uy);
	cvReleaseImage(&devx);
	cvReleaseImage(&devy);
	cvReleaseImage(&covxy);

	return SSIM;
}

IplImage* dev(IplImage* src,IplImage* avg,double sigma)
{
	double c=0;
	double mask[2*SN+1][2*SN+1];
	if(sigma!=0)
	{
		for(int i=-SN;i<=SN;++i)
			for(int j=-SN;j<=SN;++j)
				c+=exp(-(i*i+j*j)/(2*sigma*sigma));

		for(int i=0;i<=2*SN;++i)
			for(int j=0;j<=2*SN;++j)
				mask[i][j]=exp(-((i-SN)*(i-SN)+(j-SN)*(j-SN))/(2*sigma*sigma))/c;
	}
	else
	{
		c=SN*SN;
		for(int i=0;i<=2*SN;++i)
			for(int j=0;j<=2*SN;++j)
				mask[i][j]=1/c;
	}

	BwImageFloat psrc(src);
	BwImageFloat pavg(avg);
	IplImage* d=cvCreateImage(cvSize(src->width,src->height),IPL_DEPTH_32F,1);
	BwImageFloat pd(d);
	for(int i=SN;i<d->height-SN;i++)
		for(int j=SN;j<d->width-SN;j++)
		{
			float f=0;
			for(int x=-SN;x<=SN;x++)
				for(int y=-SN;y<=SN;y++)
					f+=mask[x+SN][y+SN]*(psrc[i+x][j+y]-pavg[i][j])*(psrc[i+x][j+y]-pavg[i][j]);
			pd[i][j]=cvSqrt(f);
		}
		return d;
}

IplImage* cov(IplImage* srcref,IplImage* avgref,IplImage* srcdis,IplImage* avgdis,double sigma)
{
	double c=0;
	double mask[2*SN+1][2*SN+1];
	if(sigma!=0)
	{
		for(int i=-SN;i<=SN;++i)
			for(int j=-SN;j<=SN;++j)
				c+=exp(-(i*i+j*j)/(2*sigma*sigma));

		for(int i=0;i<=2*SN;++i)
			for(int j=0;j<=2*SN;++j)
				mask[i][j]=exp(-((i-SN)*(i-SN)+(j-SN)*(j-SN))/(2*sigma*sigma))/c;
	}
	else
	{
		c=SN*SN;
		for(int i=0;i<=2*SN;++i)
			for(int j=0;j<=2*SN;++j)
				mask[i][j]=1/c;
	}

	BwImageFloat psrcref(srcref);
	BwImageFloat psrcdis(srcdis);
	BwImageFloat pavgref(avgref);
	BwImageFloat pavgdis(avgdis);
	IplImage* covxy=cvCreateImage(cvSize(srcref->width,srcref->height),IPL_DEPTH_32F,1);
	cvZero(covxy);
	BwImageFloat pcovxy(covxy);
	for(int i=SN;i<covxy->height-SN;i++)
		for(int j=SN;j<covxy->width-SN;j++)
			for(int x=-SN;x<=SN;x++)
				for(int y=-SN;y<=SN;y++)
					pcovxy[i][j]+=mask[x+SN][y+SN]*(psrcref[i+x][j+y]-pavgref[i][j])*(psrcdis[i+x][j+y]-pavgdis[i][j]);

	return covxy;
}

