#include "globals.cpp"
#include <bitset>
#include <map>
#include <set>
#include <vector>

// Function declarations
void InitializeEnvironment();
void OutputEdgeFailureScenarios();
void GenerateFailureScenarios();
void HandleTestScenarios();
void ResetEnvironment();
void HandleEdgeFailure(int e);
std::vector<int> DetectAffectedServices(int e);
std::vector<int>
AssignWavelengthsUsingGA(const std::vector<int> &affected_services);
bool RunGAWithTemp(Service &srv,
                   std::vector<std::bitset<MAX_K + 1>> &temp_edge_wavelengths,
                   std::vector<int> &temp_Pi);
double FitnessFunction(
    const std::vector<int> &chromosome, const Service &srv,
    const std::vector<std::bitset<MAX_K + 1>> &temp_edge_wavelengths,
    const std::vector<int> &temp_Pi);
std::vector<int> GenerateInitialChromosome(const Service &srv);
std::vector<int> MutateChromosome(const std::vector<int> &chromosome,
                                  const Service &srv);
std::pair<std::vector<int>, std::vector<int>>
CrossoverChromosomes(const std::vector<int> &parent1,
                     const std::vector<int> &parent2, const Service &srv);
bool DecodeChromosomeWithTemp(
    const std::vector<int> &chromosome, const Service &srv,
    std::vector<int> &path, std::vector<int> &wavelengths,
    std::map<int, int> &converters_needed,
    const std::vector<std::bitset<MAX_K + 1>> &temp_edge_wavelengths,
    const std::vector<int> &temp_Pi);
std::vector<std::pair<int, int>> GetAvailableIntervalsWithTemp(
    int edge_id, int width, const Service &srv,
    const std::vector<std::bitset<MAX_K + 1>> &temp_edge_wavelengths);
double JaccardSimilarity(const std::set<int> &a, const std::set<int> &b);
std::vector<int> ComputeMinCut(int s, int d);
