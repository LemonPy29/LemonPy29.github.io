---
layout: page
title: Coding Questions
nav: false
---
<link rel="stylesheet" href="/assets/css/main.css"/>

### Obstacle Paths [Twitch]
*Statment* : Given a 2-d array of 0 and 1, determine how many path exists from the top left corner to the bottom right corner given that at any point the only moves allowed
are right and down.

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
                elif i>= 1 and j >= 1:
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1] 

    return dp[-1][-1]
```
