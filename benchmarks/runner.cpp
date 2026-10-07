#define VECTORDB_NO_MAIN
#include "../main.cpp"
#include <sys/resource.h>
using Clock=std::chrono::steady_clock;
double ms(Clock::time_point start){return std::chrono::duration<double,std::milli>(Clock::now()-start).count();}
int main(int argc,char** argv){
 int n=argc>1?std::stoi(argv[1]):1000,d=argc>2?std::stoi(argv[2]):16,queries=argc>3?std::stoi(argv[3]):100,m=argc>4?std::stoi(argv[4]):16,buildEf=argc>5?std::stoi(argv[5]):200;
 std::string metric=argc>6?argv[6]:"cosine";auto distance=getDistFn(metric);
 if(n<10||d<1||queries<1)throw std::invalid_argument("n>=10, dims>=1, queries>=1 required");
 std::mt19937 rng(20261007);std::normal_distribution<float> normal;
 auto vector=[&]{std::vector<float> v(d);for(auto& x:v)x=normal(rng);return v;};
 BruteForce bf;HNSW h(m,buildEf);auto start=Clock::now();
 for(int id=0;id<n;id++){VectorItem item{id,"","",vector()};bf.insert(item);h.insert(item,distance);}
 double buildMs=ms(start);struct rusage usage{};getrusage(RUSAGE_SELF,&usage);
 std::vector<std::vector<float>> qs;for(int i=0;i<queries;i++)qs.push_back(vector());
 std::cout<<"n,dims,queries,seed,metric,M,efConstruction,efSearch,k,recall,exact_mean_us,hnsw_mean_us,hnsw_p50_us,hnsw_p95_us,build_ms,process_peak_rss_kib\n";
 for(int k:{1,5,10}){
  std::vector<std::vector<std::pair<float,int>>> ground;double exactUs=0;
  for(auto& q:qs){start=Clock::now();ground.push_back(bf.knn(q,k,distance));exactUs+=ms(start)*1000;}
  for(int ef:{10,50,100,200}){
   h.knn(qs[0],k,ef,distance); // untimed warm-up
   std::vector<double> latency;double recall=0;
   for(int i=0;i<queries;i++){start=Clock::now();auto hits=h.knn(qs[i],k,ef,distance);latency.push_back(ms(start)*1000);
    std::set<int> ids;for(auto& x:ground[i])ids.insert(x.second);int matches=0;for(auto& x:hits)matches+=ids.count(x.second);recall+=double(matches)/ground[i].size();}
   double total=0;for(double x:latency)total+=x;std::sort(latency.begin(),latency.end());
   std::cout<<n<<','<<d<<','<<queries<<",20261007,"<<metric<<','<<m<<','<<buildEf<<','<<ef<<','<<k<<','<<recall/queries<<','<<exactUs/queries<<','<<total/queries<<','<<latency[(queries-1)/2]<<','<<latency[size_t(std::ceil(.95*queries))-1]<<','<<buildMs<<','<<usage.ru_maxrss<<'\n';
  }
 }
}
