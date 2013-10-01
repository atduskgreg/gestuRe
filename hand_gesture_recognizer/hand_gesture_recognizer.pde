import processing.video.*;

import gab.opencv.*;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.TermCriteria;

import hog.*;


import java.util.Map;
import java.util.Arrays;

RandomForest classifier;
OpenCV opencv;

Capture video;

ArrayList<Sample> trainingSamples;

int w = 640;
int h = 480;

PImage testImage;
int rectW = 150;
int rectH = 150;

void setup(){
  opencv = new OpenCV(this,0,0);
  classifier = new RandomForest(this);
  
  video = new Capture(this, w/2, h/2);
  video.start();
  
  size(w/2 + 60,h/2);
  
  trainingSamples = new ArrayList<Sample>();
  
  testImage = createImage(50, 50, RGB);
}

void draw() {
  background(0);
  image(video,0,0);
  noFill();
  stroke(255,0,0);
  strokeWeight(5);
  rect(video.width - rectW - (video.width - rectW)/2, video.height - rectH - (video.height - rectH)/2, rectW, rectH);
  
  testImage.copy(video, video.width - rectW - (video.width - rectW)/2, video.height - rectH - (video.height - rectH)/2, rectW, rectH, 0, 0, 50, 50);
  
  image(testImage, width-testImage.width, 0);
  
}

void captureEvent(Capture c) {
  c.read();
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
