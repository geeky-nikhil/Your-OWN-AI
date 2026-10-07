#define VECTORDB_NO_MAIN
#include "../main.cpp"
#include <iostream>
int main(){
 std::mt19937 rng(42);std::normal_distribution<float> normal;std::uniform_real_distribution<float> scale(.1,10);
 auto sample=[&]{std::vector<float> v(16);float s=scale(rng);for(auto& x:v)x=normal(rng)*s;return v;};
 std::vector<VectorItem> data;for(int i=0;i<1000;i++)data.push_back({i,"","",sample()});
 std::vector<std::vector<float>> queries;for(int i=0;i<100;i++)queries.push_back(sample());
 std::cout<<"metric,n,queries,k,efSearch,cosine_built_recall,metric_consistent_recall\n";
 for(auto metric:{"euclidean","manhattan"}){
  auto dist=getDistFn(metric);HNSW legacy,correct;BruteForce bf;
  for(auto& v:data){legacy.insert(v,cosine);correct.insert(v,dist);bf.insert(v);}
  double oldRecall=0,newRecall=0;
  for(auto& q:queries){std::set<int> ids;for(auto& p:bf.knn(q,10,dist))ids.insert(p.second);
   for(auto& p:legacy.knn(q,10,50,dist))oldRecall+=ids.count(p.second)/10.;
   for(auto& p:correct.knn(q,10,50,dist))newRecall+=ids.count(p.second)/10.;}
  std::cout<<metric<<",1000,100,10,50,"<<oldRecall/100<<','<<newRecall/100<<'\n';
 }
}
