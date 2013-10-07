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

ArrayList<ArrayList<PImage>> classImages;

ArrayList<Sample> trainingSamples;

int w = 640;
int h = 480;

PImage testImage;
int rectW = 150;
int rectH = 150;

int currentLabel = 0;
boolean trained = false;

int numClasses = 5;

PImage defaultImage;

void setup() {
  opencv = new OpenCV(this, 50, 50);
  classifier = new Libsvm(this);
  classifier.setNumFeatures(1728);

  video = new Capture(this, w/2, h/2);
  video.start();

  size(1000, 600);

  trainingSamples = new ArrayList<Sample>();

  classImages = new ArrayList<ArrayList<PImage>>();
  for(int i = 0; i < numClasses; i++){
    classImages.add(new ArrayList<PImage>());
  }

  testImage = createImage(50, 50, RGB);
  defaultImage = createImage(50,50,RGB);
  for(int i = 0; i < defaultImage.pixels.length; i++){
    defaultImage.pixels[i] = color(0,255,0);
  }
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


    TreeMap<Double, PImage> map = new TreeMap<Double, PImage>();
    for ( int i = 0; i < confidence.length; i++ ) {
      if(classImages.get(i).size() > 0){
        map.put( confidence[i], classImages.get(i).get(0) );
      } else {
        map.put( confidence[i], defaultImage );
      }
    }


    //Arrays.sort(indexes, comparator);
    text("label: " + prediction, w/2+10, 60);
    image(classImages.get((int)prediction).get(0), w/2+ 70, 0);

    //String report = "CONF\n";
    String[] report = {
      "", "", "", "", ""
    };

    int i = 0;
    //for (int i = 0; i < 5; i++) {
      pushMatrix();
      translate(w/2+10, 75);
    for (Map.Entry entry : map.entrySet() ) {
      image((PImage)entry.getValue(),0,0, 20,20);
      translate(0, 25);
      double k = (Double)entry.getKey();
      report[i] = nfc((float)k, 2) + "\n\n";
      i++;
    }
    popMatrix();
    text("CONF\n" + join(report, ""), w/2+35, 75);
  }

  text("(t)rain", w/2+10, 160);

  text("(a)dd\nlabel\nto: " + currentLabel + "\n(n)next", w/2+10, height- 50);

  image(testImage, w/2+ 10, 0);
  
  
  pushMatrix();
  translate(w/2 + 160, 0);
  for(int i = 0; i < classImages.size(); i++){
    pushMatrix();
    translate(i*50, 0);
    text("C" + i, 0,15);
    ArrayList<PImage> images = classImages.get(i);
    for(int j = 0; j < images.size(); j++){
      image(images.get(j), 0, 50*j + 25);
    }
    popMatrix();
  }
  popMatrix();
  
}

void keyPressed() {
  if (key == 'n') {
    currentLabel++;
    if (currentLabel > 5) {
      currentLabel = 0;
    }
  }

  if (key == 'a') {
    classifier.addTrainingSample( new Sample(gradientsForImage( testImage, currentLabel ), currentLabel) );
  }

  if (key == 't') {
    classifier.train();
    trained = true;
  }
}

void captureEvent(Capture c) {
  c.read();
}

float[] gradientsForImage(PImage img, int label) {
  img.resize(50,50);
  img.updatePixels();
  PImage labeledImage = createImage(50,50, RGB);
  labeledImage.copy(img, 0,0, 50,50, 0,0,50,50);
  labeledImage.updatePixels();
  classImages.get(label).add(labeledImage);
  return gradientsForImage(img);
}

float[] gradientsForImage(PImage img) {
  // resize the images to a consistent size:
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

