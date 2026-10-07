#define VECTORDB_NO_MAIN
#include "../main.cpp"
#include <cassert>
#include <iostream>
template<class F> void rejects(F fn) {bool ok=false;try{fn();}catch(const std::exception&){ok=true;}assert(ok);}
int main(){
 rejects([]{parseVec("1oops,2");}); rejects([]{parseVec("nan,2");}); rejects([]{parseVec("1,2,");});
 assert(chunkText("a b c d e f",4,1)==std::vector<std::string>({"a b c d","d e f"}));
 rejects([]{chunkText("a",3,3);}); rejects([]{HNSW h(1,10);});
 VectorDB db(4); std::mt19937 rng(123);std::normal_distribution<float> normal;
 for(int i=0;i<300;i++){std::vector<float> v(4);for(auto& x:v)x=normal(rng);db.insert("x","test",v,cosine);}
 for(auto metric:{"cosine","euclidean","manhattan"})for(int i=0;i<30;i++){
   std::vector<float> q(4);for(auto& x:q)x=normal(rng);
   auto exact=db.search(q,10,metric,"bruteforce");auto kd=db.search(q,10,metric,"kdtree");
   assert(exact.hits.size()==kd.hits.size());for(int j=0;j<10;j++)assert(exact.hits[j].id==kd.hits[j].id);
   assert(db.benchmark(q,10,metric,300).hnswRecallAtK>=0.8);
 }
 rejects([&]{db.search({1,2},3,"cosine","hnsw");});
 rejects([&]{db.search({1,2,3,4},0,"cosine","hnsw");});
 rejects([&]{db.search({1,2,3,4},2,"invalid","hnsw");});
 rejects([&]{db.insert("x","y",{NAN,1,2,3},cosine);});
 auto dir=std::filesystem::temp_directory_path()/"vector-regression";std::filesystem::remove_all(dir);std::filesystem::create_directories(dir);
 auto file=(dir/"vectors").string();int id;
 {VectorDB v(2,16,200,file);id=v.insert("quote \" and\n newline","x",{1,2},cosine);v.insert("b","y",{2,1},cosine);}
 {VectorDB v(2,16,200,file);assert(v.size()==2);assert(v.all()[0].emb.size()==2);v.remove(id);}
 {VectorDB v(2,16,200,file);assert(v.size()==1);assert(v.insert("c","z",{3,2},cosine)==3);}
 auto docs=(dir/"docs").string();
 {DocumentDB d(16,200,docs);d.insert("source","the answer is 42",{1,0});rejects([&]{d.insert("bad","bad",{1,2,3});});}
 {DocumentDB d(16,200,docs);assert(d.size()==1);assert(d.search({1,0},3)[0].second.text=="the answer is 42");assert(d.search({0,1},3,0.1).empty());d.remove(1);}
 {DocumentDB d(16,200,docs);assert(d.size()==0);d.insert("new","new",{1,2,3});}
 HNSW graph;for(int i=1;i<=30;i++)graph.insert({i,"","",{float(i),1}},euclidean);
 auto info=graph.getInfo();int top=-1;for(auto& n:info.nodes)if(n.maxLyr==info.topLayer)top=n.id;
 graph.remove(top);info=graph.getInfo();int highest=-1;for(auto& n:info.nodes)highest=std::max(highest,n.maxLyr);assert(highest==info.topLayer);
 for(auto hit:graph.knn({10,1},10,50,euclidean))assert(hit.second!=top);
 std::ofstream(file)<<"VECTORDB 1 3 1\n1 broken";rejects([&]{VectorDB v(2,16,200,file);});
 std::filesystem::remove_all(dir);std::cout<<"Core regression checks passed\n";
}
