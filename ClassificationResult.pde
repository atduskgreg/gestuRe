class ClassificationResult {
  int falsePositive = 0;
  int falseNegative = 0;
  int truePositive = 0;
  int trueNegative = 0;

  ClassificationResult() {}
  
  void addResult(boolean positive, boolean correct){
    if(positive){
     if(correct){
       truePositive++;
     } else {
       falsePositive++;
     }
    } else {
      if(correct){
        trueNegative++;
      } else {
        falseNegative++;
      }
    }
  }
  
  float getAccuracy(){
    return (float)(truePositive + trueNegative)/(truePositive + trueNegative + falsePositive + falseNegative);
  }
  
  float getPrecision(){
    return (float)truePositive/(truePositive + falsePositive);
  }
  
  float getRecall(){
    return (float)truePositive/(truePositive + falseNegative);
  }
 
  float getFMeasure(){
    return 2*((getPrecision()*getRecall())/(getPrecision()+getRecall()));
  }
  
}
