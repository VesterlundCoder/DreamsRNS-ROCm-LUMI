#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include "rns/topk.h"

using namespace rns;

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST_ASSERT(cond, msg) do { \
  if (!(cond)) { \
    std::cerr << "FAIL: " << msg << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
    tests_failed++; \
    return; \
  } \
} while(0)

#define TEST_PASS() tests_passed++

void test_topk_ascending() {
  int B = 100;
  int Kkeep = 10;
  
  std::vector<float> scores(B);
  std::vector<float> est(B);
  
  for (int i = 0; i < B; ++i) {
    scores[i] = (float)(B - i);  // [100, 99, 98, ..., 1]
    est[i] = scores[i] * 2.0f;
  }
  
  std::vector<TopKItem> topk(Kkeep);
  
  TopKConfig cfg;
  cfg.B = B;
  cfg.Kkeep = Kkeep;
  cfg.ascending = true;
  
  topk_reduce_cpu(scores.data(), est.data(), topk.data(), cfg);
  
  // Smallest 10 should be [1, 2, ..., 10]
  for (int i = 0; i < Kkeep; ++i) {
    TEST_ASSERT(topk[i].score == (float)(i + 1), "Correct score");
    TEST_ASSERT(topk[i].shift_idx == B - 1 - i, "Correct index");
  }
  
  TEST_PASS();
  std::cout << "  test_topk_ascending: PASS" << std::endl;
}

void test_topk_descending() {
  int B = 100;
  int Kkeep = 10;
  
  std::vector<float> scores(B);
  std::vector<float> est(B);
  
  for (int i = 0; i < B; ++i) {
    scores[i] = (float)(i + 1);  // [1, 2, ..., 100]
    est[i] = scores[i] * 0.5f;
  }
  
  std::vector<TopKItem> topk(Kkeep);
  
  TopKConfig cfg;
  cfg.B = B;
  cfg.Kkeep = Kkeep;
  cfg.ascending = false;
  
  topk_reduce_cpu(scores.data(), est.data(), topk.data(), cfg);
  
  // Largest 10 should be [100, 99, ..., 91]
  for (int i = 0; i < Kkeep; ++i) {
    TEST_ASSERT(topk[i].score == (float)(B - i), "Correct score");
    TEST_ASSERT(topk[i].shift_idx == B - 1 - i, "Correct index");
  }
  
  TEST_PASS();
  std::cout << "  test_topk_descending: PASS" << std::endl;
}

void test_topk_random() {
  int B = 1000;
  int Kkeep = 20;
  
  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> dist(0.0f, 1000.0f);
  
  std::vector<float> scores(B);
  std::vector<float> est(B);
  
  for (int i = 0; i < B; ++i) {
    scores[i] = dist(rng);
    est[i] = dist(rng);
  }
  
  std::vector<TopKItem> topk(Kkeep);
  
  TopKConfig cfg;
  cfg.B = B;
  cfg.Kkeep = Kkeep;
  cfg.ascending = true;
  
  topk_reduce_cpu(scores.data(), est.data(), topk.data(), cfg);
  
  // Verify sorted order
  for (int i = 1; i < Kkeep; ++i) {
    TEST_ASSERT(topk[i].score >= topk[i-1].score, "Sorted ascending");
  }
  
  // Verify these are actually the smallest
  std::vector<std::pair<float, int>> sorted_scores(B);
  for (int i = 0; i < B; ++i) {
    sorted_scores[i] = {scores[i], i};
  }
  std::sort(sorted_scores.begin(), sorted_scores.end());
  
  for (int i = 0; i < Kkeep; ++i) {
    TEST_ASSERT(topk[i].score == sorted_scores[i].first, "Matches sorted");
  }
  
  TEST_PASS();
  std::cout << "  test_topk_random: PASS" << std::endl;
}

void test_topk_preserves_est() {
  int B = 50;
  int Kkeep = 5;
  
  std::vector<float> scores(B);
  std::vector<float> est(B);
  
  for (int i = 0; i < B; ++i) {
    scores[i] = (float)i;
    est[i] = (float)(i * 100);  // distinct values
  }
  
  std::vector<TopKItem> topk(Kkeep);
  
  TopKConfig cfg;
  cfg.B = B;
  cfg.Kkeep = Kkeep;
  cfg.ascending = true;
  
  topk_reduce_cpu(scores.data(), est.data(), topk.data(), cfg);
  
  for (int i = 0; i < Kkeep; ++i) {
    int idx = topk[i].shift_idx;
    TEST_ASSERT(topk[i].est == est[idx], "Est preserved");
  }
  
  TEST_PASS();
  std::cout << "  test_topk_preserves_est: PASS" << std::endl;
}

int main() {
  std::cout << "=== TopK Tests ===" << std::endl;
  
  test_topk_ascending();
  test_topk_descending();
  test_topk_random();
  test_topk_preserves_est();
  
  std::cout << "\n=== Summary ===" << std::endl;
  std::cout << "Passed: " << tests_passed << std::endl;
  std::cout << "Failed: " << tests_failed << std::endl;
  
  return tests_failed > 0 ? 1 : 0;
}
