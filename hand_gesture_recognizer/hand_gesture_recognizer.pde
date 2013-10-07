import processing.video.*;

import gab.opencv.*;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.TermCriteria;

import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfPoint;

import org.opencv.objdetect.HOGDescriptor;

import org.opencv.core.Size;
import org.opencv.core.Scalar;
import org.opencv.core.Core;

import java.util.Map;
import java.util.Arrays;
import java.util.TreeMap;

Libsvm classifier;
OpenCV opencv;

Capture video;

ArrayList<Sample> trainingSamples;

int w = 640;
int h = 480;

PImage testImage;
int rectW = 150;
int rectH = 150;

int currentLabel = 0;
boolean trained = false;

void setup() {
  opencv = new OpenCV(this, 50, 50);
  classifier = new Libsvm(this);
  classifier.setNumFeatures(1728);

  video = new Capture(this, w/2, h/2);
  video.start();

  size(w, h/2);

  trainingSamples = new ArrayList<Sample>();

  testImage = createImage(50, 50, RGB);
}

void draw() {
  background(0);
  image(video, 0, 0);
  noFill();
  stroke(255, 0, 0);
  strokeWeight(5);
  rect(video.width - rectW - (video.width - rectW)/2, video.height - rectH - (video.height - rectH)/2, rectW, rectH);

  testImage.copy(video, video.width - rectW - (video.width - rectW)/2, video.height - rectH - (video.height - rectH)/2, rectW, rectH, 0, 0, 50, 50);

  if (trained) {
    double[] confidence = new double[5];
    double prediction = classifier.predict( new Sample(gradientsForImage(testImage )), confidence);


    TreeMap<Double, Integer> map = new TreeMap<Double, Integer>();
    for ( int i = 0; i < confidence.length; i++ ) {
      map.put( confidence[i], i );
    }


    //Arrays.sort(indexes, comparator);
    text("label: " + prediction, w/2+10, 60);

    //String report = "CONF\n";
    String[] report = {
      "", "", "", "", ""
    };

    int i = 4;
    //for (int i = 0; i < 5; i++) {
    for (Map.Entry entry : map.entrySet() ) {
      double k = (Double)entry.getKey();
      report[i] = entry.getValue() + ": " + nfc((float)k, 2) + "\n";
      i--;
    }
    text("CONF\n" + join(report, ""), w/2+10, 75);
  }

  text("(t)rain", w/2+10, 160);

  text("(a)dd\nlabel\nto: " + currentLabel + "\n(n)next", w/2+10, height- 50);


  image(testImage, w/2+ 10, 0);
}

void keyPressed() {
  if (key == 'n') {
    currentLabel++;
    if (currentLabel > 5) {
      currentLabel = 0;
    }
  }

  if (key == 'a') {
    classifier.addTrainingSample( new Sample(gradientsForImage( testImage ), currentLabel) );
  }

  if (key == 't') {
    classifier.train();
    trained = true;
  }
}

void captureEvent(Capture c) {
  c.read();
}

float[] gradientsForImage(PImage img) {
  // resize the images to a consistent size:
  img.resize(50, 50);
  opencv.loadImage(img);

  Mat angleMat, gradMat;
  Size winSize = new Size(40, 24);
  Size blockSize = new Size(8, 8);
  Size blockStride = new Size(16, 16);
  Size cellSize = new Size(2, 2);
  int nBins = 9;
  Size winStride = new Size(16, 16);
  Size padding = new Size(0, 0);

  HOGDescriptor descriptor = new HOGDescriptor(winSize, blockSize, blockStride, cellSize, nBins);

  MatOfFloat descriptors = new MatOfFloat();

  //descriptor.compute(opencv.getGray(), descriptors);
  //Size winStride, Size padding, MatOfPoint locations
  MatOfPoint locations = new MatOfPoint();
  descriptor.compute(opencv.getGray(), descriptors, winStride, padding, locations);

  return descriptors.toArray();
}

