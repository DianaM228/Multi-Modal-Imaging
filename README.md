# Multi-Modal-Imaging

In many scientific fields, multiple imaging modalities are employed to capture distinct features of an object, thereby enhancing our understanding of the underlying processes. This multifaceted approach underscores the need for innovative methods in multimodal information analysis within the realm of image analysis.

As a practical case study, the developed methods were tested on Mass Cytometry Imaging (IMC). IMC is a cutting-edge technology that generates high-content spatial information, enabling the study of immune cell mechanisms within digital pathology. Traditionally, the analysis of IMC data has relied on segmenting individual cells. While effective, these data-driven approaches are limited, and comprehensive investigations leveraging the full spectrum of available information remain scarce.

In this repository, we propose two multimodal image fusion pipelines designed to process raw IMC data. These pipelines open new avenues for extracting valuable insights from this advanced imaging modality and facilitate the evaluation of novel multimodal imaging analysis techniques.

The first method is based on an autoencoder and the second on a multi-head model for intermediate image fusion, the latter based on the multimodal model originally proposed in [1]

### References
1. Vale-Silva, L.A., Rohr, K. Long-term cancer survival prediction using multimodal deep learning. Sci Rep 11, 13505 (2021) [https://rdcu.be/dViM4]
