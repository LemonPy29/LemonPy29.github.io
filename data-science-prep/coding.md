---
layout: page
title: Coding Questions
nav: false
---
<link rel="stylesheet" href="/assets/css/main.css"/>

### Obstacle Paths [Twitch]
*Statment* : Given a 2-d array of 0 and 1, determine how many path exists from the top left corner to the bottom right corner given that at any point the only moves allowed are right and down.

```python
def obstacles(grid):
    m = len(grid)
    n = len(grid[0])
    
    dp = [[0 for j in range(n)] for i in range(m)]
    
    dp[0][1] = 1 if grid[0][1] == 0 else 0
    dp[1][0] = 1 if grid[1][0] == 0 else 0
    
    for i in range(m):
        for j in range(n): 
            if grid[i][j] != 1:
                if i == 0 and j >= 2:
                    dp[i][j] = dp[i][j - 1] 
                elif i >= 2 and j == 0:
                    dp[i][j] = dp[i - 1][j] 
                elif i >= 1 and j >= 1:
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1] 

    return dp[-1][-1]
```

### Max of Sliding Window [Lyft]
*Statment*: Given an array A of positive integers and an integer k, write a function to get the largest value within the sliding window of size k for A. Each sliding window is k numbers and moves from the leftmost to the rightmost within A, one position at a time.

For example, if A = [2, 5, 3, 1, 4] and n = 2, then you should return [5, 5, 3, 4].

```python
# Time: O(k * n)
# Space: O(k * n)

def max_sliding_window(a: list, k: int):
    l = len(a)
    dp = [[0] * (l-i) for i in range(k)]
    dp[0] = a
    
    for i in range(1, k):
        for j in range(l-i):
            dp[i][j] = max(dp[i - 1][j], dp[i - 1][j + 1])
    
    return dp[-1]
```
