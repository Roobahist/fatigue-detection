# Selection and provenance of the displayed figures

The figures on [the results page](research-results.md) were checked against the active, non-commented `includegraphics` references in the research manuscript. **27 unique image paths were referenced and all 27 resolved.** Ten of those referenced image blobs already existed in this repository at alternate paths with identical Git content hashes.

This page lists **only the 11 images used by the public summary**. The three training panels are one grouped manuscript figure. Git blob hashes establish **byte-for-byte equality**, rather than matching solely on filename or image appearance. All source figures were directly referenced in the current manuscript rather than selected by creation date or by the newest filename.

| Manuscript figure | Public path | Selection | Git blob SHA-1 |
| --- | --- | --- | --- |
| `fig:model` | [video-mil-architecture.png](figures/video-mil-architecture.png) | Imported exact referenced figure | `aeca687bb01baaae212919cbeeb2120243fe6492` |
| `fig:pca_variance` | [pca-explained-variance.png](figures/pca-explained-variance.png) | Imported exact referenced figure | `b1cd3f8a3b1314d58901ed771137b36645f897c0` |
| `fig:pc1_loadings` | [pca-pc1-loadings.png](figures/pca-pc1-loadings.png) | Imported exact referenced figure | `d9be88392b6d1dceaa4e442e71e93fafbd324330` |
| `fig:rt_scatter` | [reaction_times_scatter.png](../plots/reaction_time/reaction_times_scatter.png) | Reused verified identical tracked image | `914d9719ed517c639adc44c7631e5506734f4916` |
| `fig:lapses` | [cumulative_lapses_per_session.png](../plots/reaction_time/rt_lapses_microfailures/cumulative_lapses_per_session.png) | Reused verified identical tracked image | `6c318fe1c2344f96817feadd0cbb93aec0e8f38a` |
| `fig:rt_boxplots_kss` | [rt_boxplots_kss_horizontal_order_kss_then_median_rt.png](../plots/reaction_time/kss_rt_boxplots_horizontal/rt_boxplots_kss_horizontal_order_kss_then_median_rt.png) | Reused verified identical tracked image | `d1aa5af8dc7de8df0ad0f0094f67320b1ff844f9` |
| `fig:slope_dist` | [slope_distribution.png](../plots/n2/output2/plots/slope_distribution.png) | Reused verified identical tracked image | `f151fa382cc59eac627ae9f0dfbbe4b691bb2e66` |
| `fig:group_early_late` | [group_early_late.png](../plots/n2/output2/plots/group_early_late.png) | Reused verified identical tracked image | `0d03d8de6e1f60dc9c6bde45db8d7e9e64ff3dd6` |
| `fig:training_curves_combined`, panel 0 | [training-curves-inner-fold-0.png](figures/training-curves-inner-fold-0.png) | Imported exact referenced figure | `e0075e3cd12aa92e059b3a415c0a037c84c9dab1` |
| `fig:training_curves_combined`, panel 1 | [training-curves-inner-fold-1.png](figures/training-curves-inner-fold-1.png) | Imported exact referenced figure | `40064db34510fd7fde42938b90f8a3205a5b92f5` |
| `fig:training_curves_combined`, panel 2 | [training-curves-inner-fold-2.png](figures/training-curves-inner-fold-2.png) | Imported exact referenced figure | `333d3ce06c8cafee7f631a32eb5ad756f8e171bd` |

**Scope and caution.** The current manuscript source is the authoritative *repository* reference used for this selection; it has not been cross-checked against a separately signed, submitted PDF. Earlier manuscript variants and plot versions were deliberately excluded. No manuscript source file or its full contents are published here. Dataset and analysis figure captions are paraphrased on the results page, with explicit caveats where the interpretation could be overstated.

For a separate revised or submitted thesis artifact, recheck its graphics references and actual figure appearance before representing this set as the final deposited edition.
