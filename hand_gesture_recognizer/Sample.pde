class Sample {
  double[] featureVector;
  int label;
  int recordId;

   Sample(double[] featureVector, int label) {
    this.featureVector = featureVector;
    this.label = label;
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
