\ Model model_parent1_copy
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  10 x[0] + 20 x[1]
Subject To
 c0: 5 x[0] + 8 x[1] <= 60
 c1: x[0] <= 8
 c2: x[1] <= 4
 c3: x[0] >= 0
 c4: x[1] >= 0
 R5: x[0] <= 5
Bounds
End
