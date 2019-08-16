CPP_SOURCES=$(wildcard *.cpp)
CPP_PROGS=$(patsubst %.cpp, %.o, $(CPP_SOURCES))

TARGET=search

CFLAGS= -std=c++11 -fPIC

CTHRIDPART= -I. -I/usr/local/include  -I../../craft/linux/include -I/usr -L../../craft/linux/lib64 -lSeetaFaceRecognizer2 -lSeetaFaceLandmarker2 -lSeetaFaceDetector2 -L/usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgproc  -lopencv_imgcodecs -lopencv_videoio

.PHONY:clean

all: $(TARGET)

$(TARGET): $(CPP_PROGS)
	g++ -o $@ $^ $(LABEL_CC) ${CFLAGS} ${CTHRIDPART}
	ln -s ../../craft/linux/lib64/libSeetaFaceRecognizer2.so libSeetaFaceRecognizer2.so
	ln -s ../../craft/linux/lib64/libSeetaFaceLandmarker2.so libSeetaFaceLandmarker2.so
	ln -s ../../craft/linux/lib64/libSeetaFaceDetector2.so libSeetaFaceDetector2.so
	ln -s ../../craft/linux/lib64/libseetanet2.so libseetanet2.so

$(CPP_PROGS): %.o:%.cpp
	g++ -o $@ -c $< ${CFLAGS} ${CTHRIDPART}

clean:
	rm -rf $(TARGET) $(CPP_PROGS) 



