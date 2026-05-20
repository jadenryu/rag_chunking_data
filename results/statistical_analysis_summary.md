# Statistical Analysis Summary

## Section 1: Data Summary

- Total rows (after cleaning): 8229
- Rows dropped (missing values): 0
- chunking_strategy levels: ['fixed_128', 'fixed_256', 'fixed_512', 'semantic_128', 'semantic_256', 'semantic_512']
Counts: {'fixed_128': 1400, 'fixed_256': 1400, 'fixed_512': 1400, 'semantic_128': 1400, 'semantic_256': 1400, 'semantic_512': 1229}
- query_type levels: ['comparative', 'multi-hop', 'single-hop', 'unanswerable']
Counts: {'comparative': 420, 'multi-hop': 1110, 'single-hop': 5829, 'unanswerable': 870}
- domain levels: ['finance', 'general', 'medical']
Counts: {'finance': 2600, 'general': 3000, 'medical': 2629}

### Outcome metric descriptives


| Metric             | Mean   | SD     | Min    | Max    |
| ------------------ | ------ | ------ | ------ | ------ |
| context_precision  | 0.6498 | 0.4187 | 0.0000 | 1.0000 |
| context_recall     | 0.5191 | 0.4667 | 0.0000 | 1.0000 |
| faithfulness       | 0.7793 | 0.2750 | 0.0000 | 1.0000 |
| answer_correctness | 0.6867 | 0.3489 | 0.0000 | 1.0000 |
| llm_as_judge       | 0.7932 | 0.3435 | 0.0000 | 1.0000 |


## Section 2: ANOVA Results

> **Note:** Primary model uses main effects only. The full 3-way interaction model
> produces a rank-deficient design matrix due to empty cells (medical/finance
> domains contain only single-hop queries), making interaction F-tests unreliable.
> Interaction terms are estimated separately and flagged.

### context_precision  (n=7566)


| Effect                                  | F        | df_num | df_den | p-value | η² partial | Size       | Flagged      |
| --------------------------------------- | -------- | ------ | ------ | ------- | ---------- | ---------- | ------------ |
| chunking_strategy                       | 32.282   | 5      | 7555   | <0.0001 | 0.0209     | small      |              |
| query_type                              | 312.365  | 3      | 7555   | <0.0001 | 0.1103     | medium     |              |
| domain                                  | 1163.328 | 2      | 7555   | <0.0001 | 0.2355     | large      |              |
| chunking_strategy × query_type          | 1.626    | 15     | 7530   | 0.1645  | 0.0032     | negligible | ⚠ unbalanced |
| chunking_strategy × domain              | 1.623    | 10     | 7530   | 0.1655  | 0.0022     | negligible | ⚠ unbalanced |
| query_type × domain                     | 1.627    | 6      | 7530   | 0.1643  | 0.0013     | negligible | ⚠ unbalanced |
| chunking_strategy × query_type × domain | 2.169    | 30     | 7530   | 0.0895  | 0.0086     | negligible | ⚠ unbalanced |


### context_recall  (n=6710)


| Effect                                  | F        | df_num | df_den | p-value | η² partial | Size       | Flagged      |
| --------------------------------------- | -------- | ------ | ------ | ------- | ---------- | ---------- | ------------ |
| chunking_strategy                       | 11.901   | 5      | 6700   | <0.0001 | 0.0088     | negligible |              |
| query_type                              | 108.931  | 2      | 6700   | <0.0001 | 0.0315     | small      |              |
| domain                                  | 1997.879 | 2      | 6700   | <0.0001 | 0.3736     | large      |              |
| chunking_strategy × query_type          | 1.191    | 10     | 6680   | 0.3124  | 0.0018     | negligible | ⚠ unbalanced |
| chunking_strategy × domain              | 1.191    | 10     | 6680   | 0.3124  | 0.0018     | negligible | ⚠ unbalanced |
| query_type × domain                     | 0.516    | 4      | 6680   | 0.6714  | 0.0003     | negligible | ⚠ unbalanced |
| chunking_strategy × query_type × domain | 1.588    | 20     | 6680   | 0.1899  | 0.0047     | negligible | ⚠ unbalanced |


### faithfulness  (n=7537)


| Effect                                  | F       | df_num | df_den | p-value | η² partial | Size       | Flagged      |
| --------------------------------------- | ------- | ------ | ------ | ------- | ---------- | ---------- | ------------ |
| chunking_strategy                       | 6.485   | 5      | 7526   | <0.0001 | 0.0043     | negligible |              |
| query_type                              | 39.586  | 3      | 7526   | <0.0001 | 0.0155     | small      |              |
| domain                                  | 173.498 | 2      | 7526   | <0.0001 | 0.0441     | small      |              |
| chunking_strategy × query_type          | 0.493   | 15     | 7501   | 0.7406  | 0.0010     | negligible | ⚠ unbalanced |
| chunking_strategy × domain              | 0.494   | 10     | 7501   | 0.7405  | 0.0007     | negligible | ⚠ unbalanced |
| query_type × domain                     | 0.552   | 6      | 7501   | 0.6974  | 0.0004     | negligible | ⚠ unbalanced |
| chunking_strategy × query_type × domain | 0.493   | 30     | 7501   | 0.7409  | 0.0020     | negligible | ⚠ unbalanced |


### answer_correctness  (n=7579)


| Effect                                  | F       | df_num | df_den | p-value | η² partial | Size       | Flagged      |
| --------------------------------------- | ------- | ------ | ------ | ------- | ---------- | ---------- | ------------ |
| chunking_strategy                       | 8.159   | 5      | 7568   | <0.0001 | 0.0054     | negligible |              |
| query_type                              | 107.320 | 3      | 7568   | <0.0001 | 0.0408     | small      |              |
| domain                                  | 121.390 | 2      | 7568   | <0.0001 | 0.0311     | small      |              |
| chunking_strategy × query_type          | 0.994   | 15     | 7543   | 0.4094  | 0.0020     | negligible | ⚠ unbalanced |
| chunking_strategy × domain              | 0.994   | 10     | 7543   | 0.4094  | 0.0013     | negligible | ⚠ unbalanced |
| query_type × domain                     | 0.275   | 6      | 7543   | 0.8942  | 0.0002     | negligible | ⚠ unbalanced |
| chunking_strategy × query_type × domain | 0.994   | 30     | 7543   | 0.4096  | 0.0039     | negligible | ⚠ unbalanced |


### llm_as_judge  (n=6347)


| Effect                                  | F       | df_num | df_den | p-value | η² partial | Size       | Flagged      |
| --------------------------------------- | ------- | ------ | ------ | ------- | ---------- | ---------- | ------------ |
| chunking_strategy                       | 3.627   | 5      | 6336   | 0.0028  | 0.0029     | negligible |              |
| query_type                              | 30.734  | 3      | 6336   | <0.0001 | 0.0143     | small      |              |
| domain                                  | 397.664 | 2      | 6336   | <0.0001 | 0.1115     | medium     |              |
| chunking_strategy × query_type          | 1.303   | 15     | 6311   | 0.2665  | 0.0031     | negligible | ⚠ unbalanced |
| chunking_strategy × domain              | 1.304   | 10     | 6311   | 0.2659  | 0.0021     | negligible | ⚠ unbalanced |
| query_type × domain                     | 1.314   | 6      | 6311   | 0.2622  | 0.0012     | negligible | ⚠ unbalanced |
| chunking_strategy × query_type × domain | 1.294   | 30     | 6311   | 0.2697  | 0.0061     | negligible | ⚠ unbalanced |


## Section 3: Tukey HSD Post-hoc Highlights

### context_precision × chunking_strategy


| Group 1      | Group 2      | Mean Diff | p-adj  | CI lower | CI upper | Significant |
| ------------ | ------------ | --------- | ------ | -------- | -------- | ----------- |
| fixed_128    | fixed_256    | 0.0664    | 0.0007 | 0.0198   | 0.1129   | ✓           |
| fixed_128    | fixed_512    | 0.1324    | 0.0000 | 0.0859   | 0.1790   | ✓           |
| fixed_128    | semantic_128 | 0.0114    | 0.9823 | -0.0352  | 0.0579   |             |
| fixed_128    | semantic_256 | 0.0647    | 0.0010 | 0.0182   | 0.1113   | ✓           |
| fixed_128    | semantic_512 | 0.1173    | 0.0000 | 0.0684   | 0.1662   | ✓           |
| fixed_256    | fixed_512    | 0.0661    | 0.0007 | 0.0195   | 0.1126   | ✓           |
| fixed_256    | semantic_128 | -0.0550   | 0.0099 | -0.1015  | -0.0084  | ✓           |
| fixed_256    | semantic_256 | -0.0016   | 1.0000 | -0.0481  | 0.0449   |             |
| fixed_256    | semantic_512 | 0.0509    | 0.0354 | 0.0020   | 0.0998   | ✓           |
| fixed_512    | semantic_128 | -0.1210   | 0.0000 | -0.1675  | -0.0745  | ✓           |
| fixed_512    | semantic_256 | -0.0677   | 0.0005 | -0.1142  | -0.0211  | ✓           |
| fixed_512    | semantic_512 | -0.0151   | 0.9510 | -0.0640  | 0.0338   |             |
| semantic_128 | semantic_256 | 0.0534    | 0.0138 | 0.0068   | 0.0999   | ✓           |
| semantic_128 | semantic_512 | 0.1059    | 0.0000 | 0.0570   | 0.1548   | ✓           |
| semantic_256 | semantic_512 | 0.0525    | 0.0267 | 0.0037   | 0.1014   | ✓           |


### context_precision × query_type


| Group 1     | Group 2      | Mean Diff | p-adj  | CI lower | CI upper | Significant |
| ----------- | ------------ | --------- | ------ | -------- | -------- | ----------- |
| comparative | multi-hop    | 0.0025    | 0.9995 | -0.0555  | 0.0605   |             |
| comparative | single-hop   | 0.0780    | 0.0006 | 0.0266   | 0.1294   | ✓           |
| comparative | unanswerable | -0.3723   | 0.0000 | -0.4325  | -0.3121  | ✓           |
| multi-hop   | single-hop   | 0.0755    | 0.0000 | 0.0420   | 0.1090   | ✓           |
| multi-hop   | unanswerable | -0.3748   | 0.0000 | -0.4206  | -0.3289  | ✓           |
| single-hop  | unanswerable | -0.4503   | 0.0000 | -0.4874  | -0.4131  | ✓           |


### context_precision × domain


| Group 1 | Group 2 | Mean Diff | p-adj  | CI lower | CI upper | Significant |
| ------- | ------- | --------- | ------ | -------- | -------- | ----------- |
| finance | general | 0.0912    | 0.0000 | 0.0674   | 0.1151   | ✓           |
| finance | medical | 0.4875    | 0.0000 | 0.4621   | 0.5129   | ✓           |
| general | medical | 0.3962    | 0.0000 | 0.3725   | 0.4200   | ✓           |


### context_recall × chunking_strategy


| Group 1      | Group 2      | Mean Diff | p-adj  | CI lower | CI upper | Significant |
| ------------ | ------------ | --------- | ------ | -------- | -------- | ----------- |
| fixed_128    | fixed_256    | 0.0477    | 0.1351 | -0.0075  | 0.1028   |             |
| fixed_128    | fixed_512    | 0.0589    | 0.0285 | 0.0037   | 0.1140   | ✓           |
| fixed_128    | semantic_128 | -0.0246   | 0.8002 | -0.0798  | 0.0305   |             |
| fixed_128    | semantic_256 | 0.0427    | 0.2340 | -0.0124  | 0.0979   |             |
| fixed_128    | semantic_512 | 0.1085    | 0.0000 | 0.0502   | 0.1668   | ✓           |
| fixed_256    | fixed_512    | 0.0112    | 0.9924 | -0.0440  | 0.0664   |             |
| fixed_256    | semantic_128 | -0.0723   | 0.0026 | -0.1274  | -0.0171  | ✓           |
| fixed_256    | semantic_256 | -0.0049   | 0.9999 | -0.0601  | 0.0503   |             |
| fixed_256    | semantic_512 | 0.0608    | 0.0352 | 0.0025   | 0.1191   | ✓           |
| fixed_512    | semantic_128 | -0.0835   | 0.0002 | -0.1386  | -0.0283  | ✓           |
| fixed_512    | semantic_256 | -0.0161   | 0.9615 | -0.0713  | 0.0391   |             |
| fixed_512    | semantic_512 | 0.0496    | 0.1479 | -0.0087  | 0.1079   |             |
| semantic_128 | semantic_256 | 0.0674    | 0.0067 | 0.0122   | 0.1226   | ✓           |
| semantic_128 | semantic_512 | 0.1331    | 0.0000 | 0.0748   | 0.1914   | ✓           |
| semantic_256 | semantic_512 | 0.0657    | 0.0167 | 0.0074   | 0.1241   | ✓           |


### context_recall × query_type


| Group 1     | Group 2    | Mean Diff | p-adj  | CI lower | CI upper | Significant |
| ----------- | ---------- | --------- | ------ | -------- | -------- | ----------- |
| comparative | multi-hop  | 0.1701    | 0.0000 | 0.1098   | 0.2305   | ✓           |
| comparative | single-hop | -0.1660   | 0.0000 | -0.2194  | -0.1125  | ✓           |
| multi-hop   | single-hop | -0.3361   | 0.0000 | -0.3710  | -0.3013  | ✓           |


### context_recall × domain


| Group 1 | Group 2 | Mean Diff | p-adj  | CI lower | CI upper | Significant |
| ------- | ------- | --------- | ------ | -------- | -------- | ----------- |
| finance | general | 0.6858    | 0.0000 | 0.6602   | 0.7113   | ✓           |
| finance | medical | 0.5423    | 0.0000 | 0.5173   | 0.5674   | ✓           |
| general | medical | -0.1435   | 0.0000 | -0.1689  | -0.1180  | ✓           |


### faithfulness × chunking_strategy


| Group 1      | Group 2      | Mean Diff | p-adj  | CI lower | CI upper | Significant |
| ------------ | ------------ | --------- | ------ | -------- | -------- | ----------- |
| fixed_128    | fixed_256    | 0.0368    | 0.0088 | 0.0060   | 0.0676   | ✓           |
| fixed_128    | fixed_512    | 0.0520    | 0.0000 | 0.0213   | 0.0828   | ✓           |
| fixed_128    | semantic_128 | 0.0163    | 0.6605 | -0.0145  | 0.0471   |             |
| fixed_128    | semantic_256 | 0.0387    | 0.0045 | 0.0080   | 0.0695   | ✓           |
| fixed_128    | semantic_512 | 0.0476    | 0.0004 | 0.0153   | 0.0800   | ✓           |
| fixed_256    | fixed_512    | 0.0153    | 0.7179 | -0.0155  | 0.0461   |             |
| fixed_256    | semantic_128 | -0.0205   | 0.4036 | -0.0513  | 0.0103   |             |
| fixed_256    | semantic_256 | 0.0020    | 1.0000 | -0.0288  | 0.0328   |             |
| fixed_256    | semantic_512 | 0.0109    | 0.9313 | -0.0215  | 0.0432   |             |
| fixed_512    | semantic_128 | -0.0358   | 0.0119 | -0.0665  | -0.0050  | ✓           |
| fixed_512    | semantic_256 | -0.0133   | 0.8207 | -0.0441  | 0.0175   |             |
| fixed_512    | semantic_512 | -0.0044   | 0.9988 | -0.0367  | 0.0279   |             |
| semantic_128 | semantic_256 | 0.0225    | 0.2966 | -0.0083  | 0.0532   |             |
| semantic_128 | semantic_512 | 0.0313    | 0.0633 | -0.0010  | 0.0637   |             |
| semantic_256 | semantic_512 | 0.0089    | 0.9704 | -0.0234  | 0.0412   |             |


### faithfulness × query_type


| Group 1     | Group 2      | Mean Diff | p-adj  | CI lower | CI upper | Significant |
| ----------- | ------------ | --------- | ------ | -------- | -------- | ----------- |
| comparative | multi-hop    | -0.0744   | 0.0000 | -0.1148  | -0.0341  | ✓           |
| comparative | single-hop   | -0.1136   | 0.0000 | -0.1493  | -0.0778  | ✓           |
| comparative | unanswerable | -0.0960   | 0.0000 | -0.1378  | -0.0541  | ✓           |
| multi-hop   | single-hop   | -0.0391   | 0.0001 | -0.0624  | -0.0158  | ✓           |
| multi-hop   | unanswerable | -0.0215   | 0.3060 | -0.0534  | 0.0104   |             |
| single-hop  | unanswerable | 0.0176    | 0.2973 | -0.0082  | 0.0434   |             |


### faithfulness × domain


| Group 1 | Group 2 | Mean Diff | p-adj  | CI lower | CI upper | Significant |
| ------- | ------- | --------- | ------ | -------- | -------- | ----------- |
| finance | general | 0.1306    | 0.0000 | 0.1130   | 0.1482   | ✓           |
| finance | medical | 0.0873    | 0.0000 | 0.0686   | 0.1061   | ✓           |
| general | medical | -0.0433   | 0.0000 | -0.0609  | -0.0257  | ✓           |


### answer_correctness × chunking_strategy


| Group 1      | Group 2      | Mean Diff | p-adj  | CI lower | CI upper | Significant |
| ------------ | ------------ | --------- | ------ | -------- | -------- | ----------- |
| fixed_128    | fixed_256    | 0.0336    | 0.1354 | -0.0053  | 0.0725   |             |
| fixed_128    | fixed_512    | 0.0625    | 0.0001 | 0.0236   | 0.1014   | ✓           |
| fixed_128    | semantic_128 | 0.0098    | 0.9796 | -0.0291  | 0.0487   |             |
| fixed_128    | semantic_256 | 0.0498    | 0.0036 | 0.0109   | 0.0887   | ✓           |
| fixed_128    | semantic_512 | 0.0676    | 0.0000 | 0.0268   | 0.1085   | ✓           |
| fixed_256    | fixed_512    | 0.0289    | 0.2798 | -0.0100  | 0.0678   |             |
| fixed_256    | semantic_128 | -0.0238   | 0.5026 | -0.0627  | 0.0151   |             |
| fixed_256    | semantic_256 | 0.0162    | 0.8434 | -0.0227  | 0.0551   |             |
| fixed_256    | semantic_512 | 0.0340    | 0.1663 | -0.0069  | 0.0749   |             |
| fixed_512    | semantic_128 | -0.0527   | 0.0016 | -0.0916  | -0.0138  | ✓           |
| fixed_512    | semantic_256 | -0.0127   | 0.9395 | -0.0516  | 0.0263   |             |
| fixed_512    | semantic_512 | 0.0051    | 0.9992 | -0.0357  | 0.0460   |             |
| semantic_128 | semantic_256 | 0.0400    | 0.0397 | 0.0011   | 0.0789   | ✓           |
| semantic_128 | semantic_512 | 0.0578    | 0.0008 | 0.0169   | 0.0987   | ✓           |
| semantic_256 | semantic_512 | 0.0178    | 0.8162 | -0.0231  | 0.0587   |             |


### answer_correctness × query_type


| Group 1     | Group 2      | Mean Diff | p-adj  | CI lower | CI upper | Significant |
| ----------- | ------------ | --------- | ------ | -------- | -------- | ----------- |
| comparative | multi-hop    | -0.2361   | 0.0000 | -0.2868  | -0.1854  | ✓           |
| comparative | single-hop   | -0.1571   | 0.0000 | -0.2020  | -0.1122  | ✓           |
| comparative | unanswerable | -0.2539   | 0.0000 | -0.3065  | -0.2013  | ✓           |
| multi-hop   | single-hop   | 0.0789    | 0.0000 | 0.0497   | 0.1082   | ✓           |
| multi-hop   | unanswerable | -0.0178   | 0.6636 | -0.0579  | 0.0223   |             |
| single-hop  | unanswerable | -0.0967   | 0.0000 | -0.1292  | -0.0643  | ✓           |


### answer_correctness × domain


| Group 1 | Group 2 | Mean Diff | p-adj  | CI lower | CI upper | Significant |
| ------- | ------- | --------- | ------ | -------- | -------- | ----------- |
| finance | general | 0.0684    | 0.0000 | 0.0458   | 0.0909   | ✓           |
| finance | medical | 0.1146    | 0.0000 | 0.0907   | 0.1386   | ✓           |
| general | medical | 0.0463    | 0.0000 | 0.0238   | 0.0688   | ✓           |


### llm_as_judge × chunking_strategy


| Group 1      | Group 2      | Mean Diff | p-adj  | CI lower | CI upper | Significant |
| ------------ | ------------ | --------- | ------ | -------- | -------- | ----------- |
| fixed_128    | fixed_256    | 0.0418    | 0.0490 | 0.0001   | 0.0835   | ✓           |
| fixed_128    | fixed_512    | 0.0418    | 0.0490 | 0.0001   | 0.0835   | ✓           |
| fixed_128    | semantic_128 | 0.0427    | 0.0456 | 0.0005   | 0.0850   | ✓           |
| fixed_128    | semantic_256 | 0.0559    | 0.0027 | 0.0132   | 0.0987   | ✓           |
| fixed_128    | semantic_512 | 0.0542    | 0.0041 | 0.0114   | 0.0969   | ✓           |
| fixed_256    | fixed_512    | 0.0000    | 1.0000 | -0.0417  | 0.0417   |             |
| fixed_256    | semantic_128 | 0.0009    | 1.0000 | -0.0413  | 0.0431   |             |
| fixed_256    | semantic_256 | 0.0141    | 0.9358 | -0.0286  | 0.0569   |             |
| fixed_256    | semantic_512 | 0.0124    | 0.9631 | -0.0304  | 0.0551   |             |
| fixed_512    | semantic_128 | 0.0009    | 1.0000 | -0.0413  | 0.0431   |             |
| fixed_512    | semantic_256 | 0.0141    | 0.9358 | -0.0286  | 0.0569   |             |
| fixed_512    | semantic_512 | 0.0124    | 0.9631 | -0.0304  | 0.0551   |             |
| semantic_128 | semantic_256 | 0.0132    | 0.9535 | -0.0300  | 0.0565   |             |
| semantic_128 | semantic_512 | 0.0115    | 0.9748 | -0.0318  | 0.0547   |             |
| semantic_256 | semantic_512 | -0.0018   | 1.0000 | -0.0455  | 0.0420   |             |


### llm_as_judge × query_type


| Group 1     | Group 2      | Mean Diff | p-adj  | CI lower | CI upper | Significant |
| ----------- | ------------ | --------- | ------ | -------- | -------- | ----------- |
| comparative | multi-hop    | -0.0712   | 0.0015 | -0.1214  | -0.0210  | ✓           |
| comparative | single-hop   | -0.1479   | 0.0000 | -0.1929  | -0.1029  | ✓           |
| comparative | unanswerable | -0.1258   | 0.0000 | -0.1779  | -0.0737  | ✓           |
| multi-hop   | single-hop   | -0.0767   | 0.0000 | -0.1065  | -0.0470  | ✓           |
| multi-hop   | unanswerable | -0.0546   | 0.0023 | -0.0943  | -0.0149  | ✓           |
| single-hop  | unanswerable | 0.0221    | 0.3074 | -0.0107  | 0.0549   |             |


### llm_as_judge × domain


| Group 1 | Group 2 | Mean Diff | p-adj  | CI lower | CI upper | Significant |
| ------- | ------- | --------- | ------ | -------- | -------- | ----------- |
| finance | general | 0.2574    | 0.0000 | 0.2342   | 0.2807   | ✓           |
| finance | medical | 0.2713    | 0.0000 | 0.2451   | 0.2976   | ✓           |
| general | medical | 0.0139    | 0.3336 | -0.0091  | 0.0370   |             |


## Section 4: Targeted Contrast

**Judge − Faithfulness gap: finance vs general**


|      | Finance | General |
| ---- | ------- | ------- |
| Mean | -0.1204 | 0.0252  |
| SD   | 0.5242  | 0.3646  |
| N    | 1600    | 2992    |


t(2445.9) = -9.9035, p = <0.0001, Cohen's d = -0.3225

**Interpretation:** No significant difference (t=-9.903, p=<0.0001) or unexpected direction.

## Section 5: Flagged Issues

- 0 rows dropped due to missing values in metrics or factors.
- Interaction terms involving `query_type × domain` and the three-way interaction are flagged as potentially uninterpretable due to empty cells (medical/finance queries are single-hop only).

