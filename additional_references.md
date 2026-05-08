# Additional References - Project 12 SROIE Receipt OCR + KIE

Independently sourced via CrossRef live API (2024-2026 priority). Every DOI below was hit at `https://api.crossref.org/works/{doi}` and returned HTTP 200 with matching metadata. Author / title / journal / year / DOI only; volume, issue, pages intentionally omitted.

## State-of-the-art gaps (vs current `reports/references.md`)

The existing bibliography stops in 2023 (TrOCR). The following families are entirely absent and the manuscript's "current Pareto frontier" claim cannot be defended without them:

1. **Layout-aware generative LLMs.** DocLLM (Wang et al. 2024, ACL) is the layout-aware decoder-only counterpart to LayoutLMv3 and is the right comparator when the user pivots to LLM-based extraction. LayoutLLM (Luo 2024 LREC-COLING) is the instruction-tuned sibling. Neither appears in the current refs.
2. **OCR-free MLLMs.** Donut (Kim 2022) is cited but the 2024-2025 successors that overtook it on document benchmarks are missing: mPLUG-DocOwl 1.5 (Hu 2024 EMNLP-Findings) and DocOwl2 (Hu 2025 ACL), Hierarchical Visual Feature Aggregation (Choi 2024 NeurIPS), QID (Le 2025 CVPRW). DocOwl2 specifically targets multi-page receipts and invoices, which is the German Rechnung use case the manuscript points to.
3. **2026 surveys.** "Deep learning based visually rich document content understanding: a survey" (Ding 2026 AI Review) is the canonical 2026 survey and gives the manuscript a defensible "current state" anchor that LayoutLMv3 (2022) cannot.
4. **2024+ SROIE-specific fine-tunes / benchmarks.** Sudana 2026 (LayoutLMv3 on Indonesian receipts), Avei 2024 (IPTA invoice deep learning), Bajrami 2024 (invoice benchmark study) all benchmark the same architecture family on adjacent receipt corpora and are directly comparable to the proposed pipeline.
5. **Calibration / confidence for KIE.** Rombach 2026 (IJDAR conformal prediction for KIE) is essential for the AP-automation business case, where every percent of straight-through processing claim must come with a confidence bound. The current refs contain no calibration paper.

The five gaps above are the minimum the project should cite to claim awareness of the 2024-2026 frontier.

## Architectures and models (2024-2026)

1. Wang D, Raman N, Sibue M, Ma Z, Babkin P, Kaur S. DocLLM: A Layout-Aware Generative Language Model for Multimodal Document Understanding. Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics. 2024. DOI:10.18653/v1/2024.acl-long.463
2. Hu A, Xu H, Ye J, Yan M, Zhang L, Zhang B. mPLUG-DocOwl 1.5: Unified Structure Learning for OCR-free Document Understanding. Findings of the Association for Computational Linguistics: EMNLP 2024. 2024. DOI:10.18653/v1/2024.findings-emnlp.175
3. Hu A, Xu H, Zhang L, Ye J, Yan M, Zhang J. mPLUG-DocOwl2: High-resolution Compressing for OCR-free Multi-page Document Understanding. Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics. 2025. DOI:10.18653/v1/2025.acl-long.291
4. Choi J, Han B, Park J, Park J. Hierarchical Visual Feature Aggregation for OCR-Free Document Understanding. Advances in Neural Information Processing Systems 37. 2024. DOI:10.52202/079017-3362
5. Le B, Xu S, Fu J, Huang Z, Li M, Guo Y. QID: Efficient Query-Informed ViTs in Data-Scarce Regimes for OCR-Free Visual Document Understanding. 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW). 2025. DOI:10.1109/cvprw67362.2025.00014
6. Wang J, Lin Z, Huang D, Xiong L, Jin L. LiLTv2: Language-substitutable Layout-image Transformer for Visual Information Extraction. ACM Transactions on Multimedia Computing, Communications, and Applications. 2025. DOI:10.1145/3708351
7. Luo C, Tang G, Zheng Q, Yao C, Jin L, Li C. Bi-VLDoc: bidirectional vision-language modeling for visually-rich document understanding. International Journal on Document Analysis and Recognition (IJDAR). 2025. DOI:10.1007/s10032-025-00518-w
8. Arshad A, Moetesum M, Hasan A, Shafait F. A Graph-Augmented Multi-Stage Transformer Model for Document Layout Understanding. International Journal on Document Analysis and Recognition (IJDAR). 2025. DOI:10.1007/s10032-025-00566-2

## Key information extraction methods (2024-2026)

9. Kirsch B, Allende-Cid H, Rueping S. PM3-KIE: A Probabilistic Multi-Task Meta-Model for Document Key Information Extraction. Findings of the Association for Computational Linguistics: ACL 2025. 2025. DOI:10.18653/v1/2025.findings-acl.1075
10. Majumder R, Wang Z, Yue Y, Kalita M, Liu J. Enforcing Graph Structures to Enhance Key Information Extraction in Document Analysis. Proceedings of the 20th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications. 2025. DOI:10.5220/0013240600003912
11. Hui Y, Liu J, Zhang Q, Zhou T, Song Y. SEG-Doc: A simple yet efficient graph neural network framework for document key information extraction. Neurocomputing. 2025. DOI:10.1016/j.neucom.2025.130493
12. Guo P, Song Y, Wang B, Cao Y, Ren J, Ji C. Class-Missing Semi-supervised document key information extraction via synergistic refinement estimation. Information Processing & Management. 2026. DOI:10.1016/j.ipm.2025.104335
13. Zhang C, Tu Y, Zhao Y, Yuan C, Chen H, Zhang Y. Modeling Layout Reading Order as Ordering Relations for Visually-rich Document Understanding. Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing. 2024. DOI:10.18653/v1/2024.emnlp-main.540
14. Ding Y, Vaiani L, Han C, Lee J, Garza P, Poon J. 3MVRD: Multimodal Multi-task Multi-teacher Visually-Rich Form Document Understanding. Findings of the Association for Computational Linguistics ACL 2024. 2024. DOI:10.18653/v1/2024.findings-acl.903

## Receipt and invoice applications (2024-2026)

15. Sudana O, Wirdiani A, Winama Putra A. Fine-tuned LayoutLMv3 for Indonesian receipts extraction. Bulletin of Electrical Engineering and Informatics. 2026. DOI:10.11591/eei.v15i2.10127
16. Avei U, Goularas D, Korkmaz E, Deveci B. Information Extraction from Scanned Invoice Documents Using Deep Learning Methods. 2024 IEEE Thirteenth International Conference on Image Processing Theory, Tools and Applications (IPTA). 2024. DOI:10.1109/ipta62886.2024.10755641
17. Bajrami M, Ackovska N, Stojkoska B, Lameski P, Zdravevski E. Deep Dive into Invoice Intelligence: A Benchmark Study of Leading Models for Automated Invoice Data Extraction. Lecture Notes in Networks and Systems. 2024. DOI:10.1007/978-981-97-3289-0_15
18. Kulkarni P, Deshmukh V, Rane K. Information Extraction with the Optimized Selective Kernel-Based Deep Learning from the Unstructured Invoice. New Generation Computing. 2026. DOI:10.1007/s00354-025-00311-7

## Benchmarks and datasets (2024-2026)

19. Ding Y, Han S, Li Y, Poon J. VRD-IU: Lessons from Visually Rich Document Intelligence and Understanding. Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence. 2024. DOI:10.24963/ijcai.2024/1258
20. Chen K, Chen Y, Xue Y. MosaicDoc: A Large-Scale Bilingual Benchmark for Visually Rich Document Understanding. Proceedings of the AAAI Conference on Artificial Intelligence. 2026. DOI:10.1609/aaai.v40i4.37282

## Calibration, evaluation, and surveys

21. Rombach A, Mehdiyev N. Beyond Accuracy: Understanding Model Confidence in Key Information Extraction with Conformal Prediction. International Journal on Document Analysis and Recognition (IJDAR). 2026. DOI:10.1007/s10032-026-00572-y
22. Ding Y, Han S, Lee J, Hovy E. Deep learning based visually rich document content understanding: a survey. Artificial Intelligence Review. 2026. DOI:10.1007/s10462-025-11477-3

## Verification

Each DOI above returned HTTP 200 against `https://api.crossref.org/works/{doi}` with matching author and title metadata at fetch time. Items that did not resolve, or that resolved to off-topic results (UI layout generation, brain imaging, finance ratios), were dropped rather than padded.
