class Sample {
  double[] featureVector;
  int label;
  int recordId;

  Sample(float[] floatVector, int label){    
    this.featureVector = new double[floatVector.length];
    for(int i = 0; i < floatVector.length; i++){
      this.featureVector[i] = (double)floatVector[i];
    }
    
    println(featureVector[0] + " " + floatVector[0]);
    
    this.label = label;
  }

   Sample(double[] featureVector, int label) {
    this.featureVector = featureVector;
    this.label = label;
  }
  
  // create a sample w/o a label (i.e. a test sample
  Sample(float[] floatVector){
    this.featureVector = new double[floatVector.length];
    for(int i = 0; i < floatVector.length; i++){
      this.featureVector[i] = (double)floatVector[i];
    }
  }
  
  Sample(int featureVectorSize){
    featureVector = new double[featureVectorSize];
  }
  
  void setLabel(int label){
    this.label = label;
  }
  
  void setRecordId(int recordId){
    this.recordId = recordId;
  }
  
}
