import gab.opencv.*;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.TermCriteria;

import hog.*;

import java.util.Map;
import java.util.Arrays;

Libsvm classifier;
OpenCV opencv;

int w = 640;
int h = 480;

PImage testImage;
double testResult = 0.0;

String[] trainingFilenames, testFilenames;

void setup(){
    size(200, 100); 

  opencv = new OpenCV(this,0,0);
  classifier = new Libsvm(this);
  
  
  java.io.File folder = new java.io.File(dataPath("train"));
  trainingFilenames = folder.list();  
  
  for (int i = 0; i < trainingFilenames.length; i++) {
    println("loading "  + i + "/" + trainingFilenames.length);
    String gestureLabel = split(trainingFilenames[i], '-')[0];
    int label = 0;
    if (gestureLabel.equals("A")) {
      label = 1;
    }

    if (gestureLabel.equals("B")) {
      label = 2;
    }

    if (gestureLabel.equals("C")) {
      label = 3;
    }

    if (gestureLabel.equals("V")) {
      label = 4;
    }

    if (gestureLabel.equals("Five")) {
      label = 5;
    }

    if (gestureLabel.equals("Point")) {
      label = 6;
    }
    
    float[] vector = gradientsForImage(loadImage("train/" + trainingFilenames[i]));
    Sample sample = new Sample(vector, label);
    
    classifier.addTrainingSample(sample);
  }
  classifier.setNumFeatures(324 );
  
  classifier.train();
  
  java.io.File testFolder = new java.io.File(dataPath("test"));
  testFilenames = testFolder.list();
  
  testResult = testNewImage();
}

double testNewImage() {
  // pick a random number between 0 and the number of test images
  int imgNum = (int)random(0, testFilenames.length-1);
  // load a test image
  testImage = loadImage("test/" + testFilenames[imgNum]);
  return classifier.predict(new Sample(gradientsForImage(testImage)));
}

void draw() {
  background(0);

  image(testImage, 0, 0);

  String result = "Gesture is: ";

  // display the name of the gesture
  // in a different color depending on
  // the result of our SVM test
  switch((int)testResult) {
  case 1:
    fill(255, 125, 125);
    result = result + "A";
    break;
  case 2:
    fill(125, 255, 125);
    result = result + "B";
    break;
  case 3:
    fill(125, 125, 255);
    result = result + "C";
    break;
  case 4:
    fill(125, 255, 255);
    result = result + "V";
    break;
  case 5:
    fill(255, 255, 125);
    result = result + "Five";
    break;
  case 6:
    fill(255);
    result = result + "Point";
    break;
  }


  text(result, testImage.width + 10, 20);
  
}


void keyPressed() {
  testResult = testNewImage();
}


float[] gradientsForImage(PImage img) {
  // resize the images to a consistent size:
  img.resize(50, 50);

  // settings for Histogram of Oriented Gradients
  // (probably don't change these)
  int window_width=64;
  int window_height=128;
  int bins = 9;
  int cell_size = 8;
  int block_size = 2;
  boolean signed = false;
  int overlap = 0;
  int stride=16;
  int number_of_resizes=5;

  // a bunch of unecessarily verbose HOG code
  HOG_Factory hog = HOG.createInstance();
  GradientsComputation gc=hog.createGradientsComputation();
  Voter voter=MagnitudeItselfVoter.createMagnitudeItselfVoter();
  HistogramsComputation hc=hog.createHistogramsComputation( bins, cell_size, cell_size, signed, voter);
  Norm norm=L2_Norm.createL2_Norm(0.1);
  BlocksComputation bc=hog.createBlocksComputation(block_size, block_size, overlap, norm);
  PixelGradientVector[][] pixelGradients = gc.computeGradients(img, this);
  
  hog.Histogram[][] histograms = hc.computeHistograms(pixelGradients);
  
  Block[][] blocks = bc.computeBlocks(histograms);
  Block[][] normalizedBlocks = bc.normalizeBlocks(blocks);
  DescriptorComputation dc=hog.createDescriptorComputation();    

  return dc.computeDescriptor(normalizedBlocks);
}