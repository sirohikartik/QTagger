Training initial model to find top 10 features...
Top 10 Features selected:
['io_burstiness_std_std', 'io_burstiness_std_max', 'window_start_s_std', 'file_deletion_rate_mean', 'file_deletion_rate_max', 'file_deletion_rate_min', 'time_to_first_encryption_s_std', 'window_start_s_max', 'file_deletion_rate_std', 'window_start_s_mean']

Training final model on Top 10 features...

================================================================================
FINAL RESULTS (TOP 10 FEATURES)
================================================================================

Window-level Accuracy : 0.4545

Classification Report
              precision    recall  f1-score   support

    goodware       0.41      0.46      0.44        95
  ransomware       0.50      0.45      0.47       114

    accuracy                           0.45       209
   macro avg       0.46      0.46      0.45       209
weighted avg       0.46      0.45      0.46       209

Confusion Matrix
                   Pred Goodware  Pred Ransomware
Actual Goodware               44               51
Actual Ransomware             63               51

================================================================================
PER SAMPLE RESULTS (Prob Average @ 0.40)
================================================================================
4dc06ecee904b9165fa699b026045c1b6408cc7061df3d2a7bc2b7b4f0879f4d True=ransomware  Pred=ransomware  ✓
52fc723f7e0c4202c97ac5bc2add2d1d3daa5c3f84f3d459a6a005a3ae380119 True=ransomware  Pred=goodware    ✗
5677dfad26045e271272bc98be2fd24e2f6d13737850ab1d9857fd58de05e9f9 True=ransomware  Pred=ransomware  ✓
90b06f07eb75045ea3d4ba6577afc9b58078eafeb2cdd417e2a88d7ccf0c0273 True=ransomware  Pred=ransomware  ✓
cd27a31e618fe93df37603e5ece3352a91f27671ee73bdc8ce9ad793cad72a0f True=ransomware  Pred=goodware    ✗
curl                                     True=goodware    Pred=goodware    ✓
e5834b7bdd70ec904470d541713e38fe933e96a4e49f80dbfb25148d9674f957 True=ransomware  Pred=ransomware  ✓
gpg                                      True=goodware    Pred=ransomware  ✗
rsync                                    True=goodware    Pred=goodware    ✓
sqlite                                   True=goodware    Pred=ransomware  ✗
stressng                                 True=goodware    Pred=ransomware  ✗
