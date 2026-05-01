<h3>Competition Overview</h3>
<p>

Ground-level images provide unique contextual and semantic information that complements satellite and aerial observation. Establishing reliable connections between these modalities - addressing challenges due to near-orthogonal viewpoint differences, dramatic scale and resolution disparities, and occlusion effects - unlocks a broad range of emerging applications. These include fine-grained geo-referencing of unlocalized photos, satellite data attribution, 3D scene reconstruction across varying altitudes, multi-view data fusion for digital twins, and cross-source change detection for environmental and infrastructure monitoring.

Recent advances in deep learning, cross-view retrieval, and multi-modal representation learning have markedly improved our ability to associate ground images with overhead observations; however, the task remains extremely challenging: differences in geometry, radiometry, and content visibility persist, and globally accurate ground-to-satellite localization is still limited to tens of meters without additional priors. The next frontier lies in local-context geo-localization, where approximate camera positions (e.g., within a few city blocks) enable meter-level alignment and open opportunities for downstream mapping, navigation, and situational awareness.

This cross-view geo-localization (CVGL) competition is sponsored by the Intelligence Advanced Research Projects Activity (IARPA) Walk-through Rendering from Images of Varying Altitude (WRIVA) program. It seeks to promote innovation and reproducibility for local-context ground-to-satellite image localization. Participants will receive sets of one or more satellite images and one or more ground-level images with approximate locations (e.g., within a few hundred meters) and will be evaluated on predicted camera geolocation accuracy in meters. Winning teams will be acknowledged during a session at IGARSS 2026 and may be invited to contribute to an article on the competition to be published in a special issue of Photogrammetric Engineering & Remote Sensing (PE&RS).

Data is available on [**IEEE DataPort**](https://ieee-dataport.org/competitions/wriva-cvgl-challenge-2026).

**Schedule**
* Development dataset release on DataPort: 4 April 2026
* Baseline and metrics code release on GitHub: 13 April 2026
* Development leaderboard posted on CodaBench: 1 May 2026
* Challenge dataset release on DataPort: 1 June 2026
* Challenge submission period: June 2026 – July 2026
* Selection of winning participants: August 2026
* Announcement of winners at IGARSS 2026: 9-14 August 2026

**Execution Details**
* Contest data is available for download from this DataPort repository. Development data is currently provided. Challenge data will be provided at the beginning of the challenge phase.
* Submissions will be managed with this CodaBench leaderboard.
* An example baseline solution and metrics code will be maintained on [**GitHub**](https://github.com/pubgeo/wriva-cvgl-baseline) and improved throughout the development phase. While we do not expect the baseline to be competitive, it will provide an example of using the provided development data inputs and producing outputs in the format expected for metric evaluation. Additional details about output file expectations will be provided upon leaderboard launch. A copy of the baseline code, data split pickle files, and model weights dated 4/13 is provided for download here since the non-code files are large. For the most recent code and documentation, please see the GitHub repo.

</p>