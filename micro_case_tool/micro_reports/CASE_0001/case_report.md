# Case Report: CASE_0001

## 1) Case Header

- pair: `ES-ZH`
- doc_mix: `ES + ZH docs`
- model: `bge-m3`
- method: `embed`
- doc_index_id: `mmarco-8841823-bilingual-es-zh-5bands-bge-m3`
- endpoint lambda: `0.0`
- lambda*: `0.3`
- overall delta (mixed - endpoint): `-0.019899999999999807` (CI90: [-0.40432022670832873, 0.5756559855134953])

## 2) How Many Queries Drive the Drop

- metric source counts: evaluate_perquery=1484, recomputed_from_run_qrels=0
- ΔnDCG@10 quantiles (all queries): min=-63.0930, p25=0.0000, p50=0.0000, p75=0.0000, max=100.0000
- worst-100 mean ΔnDCG@10: `-32.9215`
- control-20 mean ΔnDCG@10: `0.0000`

## 3) Failure Label Breakdown (Worst Set)

- label thresholds: mismatch_rate_mix>0.0000, endpoint_cos<0.5000, len_ratio<0.5000 or >1.5000, delta_recall<0.0000, rankdrop=(delta_ndcg<0.0000 and delta_recall>=0.0000)
- IndexLeakage: count=0, mean ΔnDCG@10=
- TranslationDivergence: count=10, mean ΔnDCG@10=-33.1719
- RecallDrop: count=33, mean ΔnDCG@10=-30.8250
- RankDrop: count=57, mean ΔnDCG@10=-34.0914
- Unclassified: count=0, mean ΔnDCG@10=

## 4) Top 20 Worst Queries

| qid | metric_source | ndcg_end | ndcg_mix | d_ndcg | rec_end | rec_mix | d_rec | first_end | first_mix | rank_shift | ov10 | ov50 | tok_a | tok_b | len_ratio | endpoint_cos | r | delta_perp | cos_to_a | cos_to_b | mismatch_end | mismatch_mix | ascii_end | ascii_mix | label |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 599720 | evaluate_perquery | 63.0930 | 0.0000 | -63.0930 | 100.0000 | 0.0000 | -100.0000 | 2 | 13 | 11.0000 | 5 | 33 | 12 | 12 | 1.0000 | 0.8009 | 0.3000 | 0.0000 | 0.9822 | 0.8990 | 0.0000 | 0.0000 | 0.6853 | 0.1267 | RecallDrop |
| 984948 | evaluate_perquery | 100.0000 | 43.0677 | -56.9323 | 100.0000 | 100.0000 | 0.0000 | 1 | 4 | 3.0000 | 8 | 39 | 9 | 8 | 1.1250 | 0.7181 | 0.3000 | 0.0000 | 0.9750 | 0.8549 | 0.0000 | 0.0000 | 0.9732 | 0.9742 | RankDrop |
| 1033927 | evaluate_perquery | 100.0000 | 50.0000 | -50.0000 | 100.0000 | 100.0000 | 0.0000 | 1 | 3 | 2.0000 | 9 | 33 | 14 | 9 | 1.5556 | 0.7672 | 0.3000 | 0.0000 | 0.9793 | 0.8812 | 0.0000 | 0.0000 | 0.9697 | 0.6799 | TranslationDivergence |
| 193422 | evaluate_perquery | 100.0000 | 50.0000 | -50.0000 | 100.0000 | 100.0000 | 0.0000 | 1 | 3 | 2.0000 | 7 | 30 | 9 | 6 | 1.5000 | 0.5986 | 0.3000 | 0.0000 | 0.9646 | 0.7886 | 0.0000 | 0.0000 | 0.7431 | 0.5888 | RankDrop |
| 408945 | evaluate_perquery | 100.0000 | 50.0000 | -50.0000 | 100.0000 | 100.0000 | 0.0000 | 1 | 3 | 2.0000 | 9 | 37 | 8 | 8 | 1.0000 | 0.7294 | 0.3000 | 0.0000 | 0.9760 | 0.8610 | 0.0000 | 0.0000 | 0.5177 | 0.4193 | RankDrop |
| 414733 | evaluate_perquery | 100.0000 | 50.0000 | -50.0000 | 100.0000 | 100.0000 | 0.0000 | 1 | 3 | 2.0000 | 10 | 45 | 12 | 10 | 1.2000 | 0.8568 | 0.3000 | 0.0000 | 0.9872 | 0.9281 | 0.0000 | 0.0000 | 0.9669 | 0.7788 | RankDrop |
| 619805 | evaluate_perquery | 100.0000 | 50.0000 | -50.0000 | 100.0000 | 100.0000 | 0.0000 | 1 | 3 | 2.0000 | 8 | 37 | 8 | 9 | 0.8889 | 0.8448 | 0.3000 | 0.0000 | 0.9861 | 0.9219 | 0.0000 | 0.0000 | 0.8619 | 0.4271 | RankDrop |
| 809798 | evaluate_perquery | 100.0000 | 50.0000 | -50.0000 | 100.0000 | 100.0000 | 0.0000 | 1 | 3 | 2.0000 | 7 | 37 | 10 | 7 | 1.4286 | 0.6763 | 0.3000 | 0.0000 | 0.9713 | 0.8320 | 0.0000 | 0.0000 | 0.9811 | 0.6988 | RankDrop |
| 337190 | evaluate_perquery | 43.0677 | 0.0000 | -43.0677 | 100.0000 | 0.0000 | -100.0000 | 4 | 20 | 16.0000 | 8 | 39 | 6 | 7 | 0.8571 | 0.8070 | 0.3000 | 0.0000 | 0.9828 | 0.9022 | 0.0000 | 0.0000 | 0.2331 | 0.0413 | RecallDrop |
| 1005949 | evaluate_perquery | 38.6853 | 0.0000 | -38.6853 | 100.0000 | 0.0000 | -100.0000 | 5 | 11 | 6.0000 | 9 | 32 | 7 | 6 | 1.1667 | 0.5243 | 0.3000 | 0.0000 | 0.9584 | 0.7457 | 0.0000 | 0.0000 | 0.9096 | 0.8627 | RecallDrop |
| 290830 | evaluate_perquery | 38.6853 | 0.0000 | -38.6853 | 100.0000 | 0.0000 | -100.0000 | 5 | 17 | 12.0000 | 7 | 40 | 7 | 9 | 0.7778 | 0.7575 | 0.3000 | 0.0000 | 0.9784 | 0.8760 | 0.0000 | 0.0000 | 0.5949 | 0.4122 | RecallDrop |
| 1000678 | evaluate_perquery | 100.0000 | 63.0930 | -36.9070 | 100.0000 | 100.0000 | 0.0000 | 1 | 2 | 1.0000 | 10 | 45 | 6 | 6 | 1.0000 | 0.7674 | 0.3000 | 0.0000 | 0.9793 | 0.8813 | 0.0000 | 0.0000 | 0.9707 | 0.9707 | RankDrop |
| 1002238 | evaluate_perquery | 100.0000 | 63.0930 | -36.9070 | 100.0000 | 100.0000 | 0.0000 | 1 | 2 | 1.0000 | 9 | 31 | 14 | 7 | 2.0000 | 0.6836 | 0.3000 | 0.0000 | 0.9720 | 0.8361 | 0.0000 | 0.0000 | 0.9711 | 0.9705 | TranslationDivergence |
| 1005653 | evaluate_perquery | 100.0000 | 63.0930 | -36.9070 | 100.0000 | 100.0000 | 0.0000 | 1 | 2 | 1.0000 | 5 | 36 | 6 | 8 | 0.7500 | 0.6764 | 0.3000 | 0.0000 | 0.9713 | 0.8321 | 0.0000 | 0.0000 | 0.7920 | 0.3017 | RankDrop |
| 1032019 | evaluate_perquery | 100.0000 | 63.0930 | -36.9070 | 100.0000 | 100.0000 | 0.0000 | 1 | 2 | 1.0000 | 7 | 38 | 10 | 7 | 1.4286 | 0.7952 | 0.3000 | 0.0000 | 0.9817 | 0.8960 | 0.0000 | 0.0000 | 0.8730 | 0.8674 | RankDrop |
| 1035874 | evaluate_perquery | 100.0000 | 63.0930 | -36.9070 | 100.0000 | 100.0000 | 0.0000 | 1 | 2 | 1.0000 | 7 | 32 | 12 | 11 | 1.0909 | 0.8108 | 0.3000 | 0.0000 | 0.9831 | 0.9042 | 0.0000 | 0.0000 | 0.9544 | 0.3188 | RankDrop |
| 1036627 | evaluate_perquery | 100.0000 | 63.0930 | -36.9070 | 100.0000 | 100.0000 | 0.0000 | 1 | 2 | 1.0000 | 6 | 31 | 6 | 6 | 1.0000 | 0.4883 | 0.3000 | 0.0000 | 0.9554 | 0.7244 | 0.0000 | 0.0000 | 0.9712 | 0.9788 | TranslationDivergence |
| 1062511 | evaluate_perquery | 100.0000 | 63.0930 | -36.9070 | 100.0000 | 100.0000 | 0.0000 | 1 | 2 | 1.0000 | 9 | 42 | 11 | 10 | 1.1000 | 0.8442 | 0.3000 | 0.0000 | 0.9861 | 0.9216 | 0.0000 | 0.0000 | 0.9812 | 0.9808 | RankDrop |
| 1082002 | evaluate_perquery | 100.0000 | 63.0930 | -36.9070 | 100.0000 | 100.0000 | 0.0000 | 1 | 2 | 1.0000 | 9 | 36 | 8 | 7 | 1.1429 | 0.8093 | 0.3000 | 0.0000 | 0.9830 | 0.9035 | 0.0000 | 0.0000 | 0.6758 | 0.3835 | RankDrop |
| 1082603 | evaluate_perquery | 100.0000 | 63.0930 | -36.9070 | 100.0000 | 100.0000 | 0.0000 | 1 | 2 | 1.0000 | 9 | 43 | 8 | 8 | 1.0000 | 0.7448 | 0.3000 | 0.0000 | 0.9773 | 0.8692 | 0.0000 | 0.0000 | 0.7323 | 0.5781 | RankDrop |

## 5) Per-Query Diff Blocks (Top 20 Worst)

All metric deltas are `mixed - endpoint` in 0-100 point units.

Note: `retrieval_score_raw` below is the original run ranking score from `.trec`, not an evaluation metric and not on the 0-100 nDCG/Recall scale.

### qid `599720`

- query A (`es`): ¿Qué complicación es un peligro potencial asociado con las infusiones intravenosas continuas?
- query B (`zh`): 与连续静脉输注相关的潜在危险是什么并发症？
- diagnosis: RecallDrop; nDCG@10 end=63.0930, mix=0.0000, Δ=-63.0930; Recall@10 end=100.0000, mix=0.0000, Δ=-100.0000; tokens(a/b)=12/12, len_ratio=1.0000; overlap@10=5; source=evaluate_perquery

Endpoint top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 1580943 | 0 | 0.6703 | zh | 获得 I.V. 的并发症可能包括浸润、血肿、空气栓塞、静脉炎、血管外给药和动脉内注射。动脉内注射更为罕见，但具有威胁性。 |
| 2 | 7466196 | 1 | 0.6584 | es | Algunos medicamentos intravenosos que se administran como infusiones a lo largo del tiempo pueden administrarse accidentalmente demasiado rápido como un ÃƒÂ ¢ Ã‚â‚¬Ã‚Å “pushÃƒÂ ... |
| 3 | 814448 | 0 | 0.6569 | zh | 4 位医生同意： 频繁：在这种危及生命的事件之后，出现并发症并不罕见，需要透析的肾功能衰竭以及长时间插管引起的肺炎在这种创伤性环境中实际上很常见。 ...阅读更多。再看 1 个医生的回答。 |
| 4 | 7466189 | 0 | 0.6521 | es | Complicaciones. Las complicaciones graves relacionadas con las vías intravenosas periféricas son poco comunes, pero ocurren problemas, especialmente con el uso prolongado. Es po... |
| 5 | 1962613 | 0 | 0.6452 | es | Sin embargo, muchas complicaciones. puede ir de la mano con la terapia intravenosa, para el examen -. ple, infiltración, flebitis, espasmo venoso, hema-. toma, embolia gaseosa y... |
| 6 | 624451 | 0 | 0.6445 | es | El riesgo de complicaciones aumenta si la úlcera no se trata o si no se completa el tratamiento. Las complicaciones pueden incluir: hemorragia interna; inestabilidad hemodinámic... |
| 7 | 4507281 | 0 | 0.6443 | es | Al igual que con todos los procedimientos médicos invasivos, existen riesgos potenciales asociados con las inyecciones epidurales de esteroides lumbares. Además del entumecimien... |
| 8 | 8244552 | 0 | 0.6443 | zh | 有哪些风险？注射的最重要风险是眼内感染。然而，这是一种非常罕见的并发症，发生在不到 1% 的注射中。其他罕见的并发症包括眼内出血、白内障、青光眼和视网膜脱离。 |
| 9 | 7376532 | 0 | 0.6437 | es | Otras complicaciones de la fluidoterapia incluyen flebitis, infiltración y extravasación. Muchos líquidos intravenosos irritan las venas, por lo que si nota enrojecimiento e hin... |
| 10 | 2357415 | 0 | 0.6422 | es | Al igual que con todos los tipos de cirugía, un injerto de derivación de arteria coronaria conlleva un riesgo de complicaciones. Por lo general, estos son relativamente leves y ... |

Mixed top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 1580943 | 0 | 0.7114 | zh | 获得 I.V. 的并发症可能包括浸润、血肿、空气栓塞、静脉炎、血管外给药和动脉内注射。动脉内注射更为罕见，但具有威胁性。 |
| 2 | 814448 | 0 | 0.7005 | zh | 4 位医生同意： 频繁：在这种危及生命的事件之后，出现并发症并不罕见，需要透析的肾功能衰竭以及长时间插管引起的肺炎在这种创伤性环境中实际上很常见。 ...阅读更多。再看 1 个医生的回答。 |
| 3 | 1962613 | 0 | 0.6939 | zh | 但是，并发症很多。可能与静脉注射疗法齐头并进，用于检查 -。 ple，浸润，静脉炎，静脉痉挛，血肿。 toma，空气栓塞，以及神经、肌腱和 liga -。精神损害。 1 其中，神经损伤是。可能很严重，因为它可能导致终生瘫痪 -。 sis、麻木和畸形。 |
| 4 | 8061386 | 0 | 0.6826 | zh | 过度灌注，或通过先前阻塞的颈动脉进入大脑动脉的血流量突然增加，可导致出血性中风。其他并发症包括再狭窄和可通过药物治疗的短期血压和心率降低。 |
| 5 | 8244552 | 0 | 0.6774 | zh | 有哪些风险？注射的最重要风险是眼内感染。然而，这是一种非常罕见的并发症，发生在不到 1% 的注射中。其他罕见的并发症包括眼内出血、白内障、青光眼和视网膜脱离。 |
| 6 | 1056079 | 0 | 0.6749 | zh | PDF：并发症。获得 I.V. 的并发症可能包括浸润、血肿、空气栓塞、静脉炎、血管外给药和动脉内注射。动脉内注射更为罕见，但具有威胁性。DF：并发症。获得 I.V. 的并发症可能包括浸润、血肿、空气栓塞、静脉炎、血管外给药和动脉内注射。动脉内注射更为罕见，但具有威胁性。 |
| 7 | 5899232 | 0 | 0.6712 | zh | 在后期，可能会出现并发症，例如感染、动脉瘤和/或假动脉瘤的形成、瘘管静脉狭窄、充血性心力衰竭、偷窃综合征、缺血性神经病变和血栓形成(表 1)。 |
| 8 | 7466189 | 0 | 0.6700 | zh | 并发症。与外周静脉注射相关的严重并发症并不常见，但确实会出现问题，尤其是长期使用时。这就是为什么不同医院都有关于外周静脉注射的推荐持续时间的指南。 |
| 9 | 958278 | 0 | 0.6642 | zh | 并发症包括。 1 中风（如果供应大脑的脑动脉被阻塞），2 心脏病发作（如果供应心肌的冠状动脉被阻塞），3 肾功能衰竭（如果供应肾脏的肾动脉，被屏蔽） |
| 10 | 8442083 | 0 | 0.6640 | es | Cuando ocurren efectos secundarios, generalmente son leves, como fiebre, náuseas, hematuria y disuria. Sin embargo, en <5% de los pacientes, la administración intravesical de BC... |

### qid `984948`

- query A (`es`): ¿Por qué me hormiguean los nervios de la nariz?
- query B (`zh`): 为什么我鼻子里的神经发麻
- diagnosis: RankDrop; nDCG@10 end=100.0000, mix=43.0677, Δ=-56.9323; Recall@10 end=100.0000, mix=100.0000, Δ=0.0000; tokens(a/b)=9/8, len_ratio=1.1250; overlap@10=8; source=evaluate_perquery

Endpoint top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 7295782 | 1 | 0.7385 | es | Hormigueo en la nariz: cualquier cosa que irrite las terminaciones nerviosas de la punta de la nariz puede provocar un hormigueo. Uno de los más comunes es la hiperventilación, ... |
| 2 | 2559828 | 0 | 0.7380 | es | La mayoría de las veces, el hormigueo en la nariz es el resultado de alergias o resfriado común. A veces, la sensación de hormigueo precede directamente a un estornudo, y otras ... |
| 3 | 6106492 | 0 | 0.7335 | es | Entumecimiento y hormigueo en la nariz. La sensación de hormigueo, entumecimiento y ardor es generalmente el síntoma cuando su nervio sensorial se daña debido a una enfermedad o... |
| 4 | 2559824 | 0 | 0.7277 | es | Entumecimiento y hormigueo en la nariz. La sensación de hormigueo, entumecimiento y ardor es generalmente el síntoma cuando su nervio sensorial se daña debido a una enfermedad o... |
| 5 | 7932934 | 0 | 0.7221 | es | El hormigueo en la nariz normalmente se debe a alergias o al resfriado común. El hormigueo nasal crónico puede ser un indicio de esclerosis múltiple. Las alergias estacionales p... |
| 6 | 2559825 | 0 | 0.7209 | es | Las siguientes condiciones médicas son algunas de las posibles causas del hormigueo en la nariz. Es probable que existan otras causas posibles, así que pregúntele a su médico ac... |
| 7 | 7932929 | 0 | 0.7197 | es | La mayoría de las veces, el hormigueo en la nariz es el resultado de alergias o resfriado común. A veces, la sensación de hormigueo precede directamente a un estornudo, y otras ... |
| 8 | 2559823 | 0 | 0.7101 | es | 1 Esto a menudo se debe a la hiperventilación, que es un síntoma común en personas con ataques de pánico. 2 Un hormigueo en la nariz puede ser una señal de que está respirando m... |
| 9 | 2434849 | 0 | 0.6971 | es | 5 formas de evitar que le haga cosquillas en la nariz. Una sensación de hormigueo en la nariz puede ser un signo de diversas afecciones, como reacción alérgica, herpes, hipopara... |
| 10 | 6263823 | 0 | 0.6918 | es | La sensación de ardor en la nariz acompañada de ardor u hormigueo en múltiples lugares del cuerpo puede ser un signo de enfermedades crónicas como la esclerosis múltiple. Si la ... |

Mixed top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 6106492 | 0 | 0.7398 | es | Entumecimiento y hormigueo en la nariz. La sensación de hormigueo, entumecimiento y ardor es generalmente el síntoma cuando su nervio sensorial se daña debido a una enfermedad o... |
| 2 | 2559824 | 0 | 0.7381 | es | Entumecimiento y hormigueo en la nariz. La sensación de hormigueo, entumecimiento y ardor es generalmente el síntoma cuando su nervio sensorial se daña debido a una enfermedad o... |
| 3 | 2559828 | 0 | 0.7340 | es | La mayoría de las veces, el hormigueo en la nariz es el resultado de alergias o resfriado común. A veces, la sensación de hormigueo precede directamente a un estornudo, y otras ... |
| 4 | 7295782 | 1 | 0.7250 | es | Hormigueo en la nariz: cualquier cosa que irrite las terminaciones nerviosas de la punta de la nariz puede provocar un hormigueo. Uno de los más comunes es la hiperventilación, ... |
| 5 | 7932929 | 0 | 0.7158 | es | La mayoría de las veces, el hormigueo en la nariz es el resultado de alergias o resfriado común. A veces, la sensación de hormigueo precede directamente a un estornudo, y otras ... |
| 6 | 7932934 | 0 | 0.7156 | es | El hormigueo en la nariz normalmente se debe a alergias o al resfriado común. El hormigueo nasal crónico puede ser un indicio de esclerosis múltiple. Las alergias estacionales p... |
| 7 | 2559825 | 0 | 0.7154 | es | Las siguientes condiciones médicas son algunas de las posibles causas del hormigueo en la nariz. Es probable que existan otras causas posibles, así que pregúntele a su médico ac... |
| 8 | 2559823 | 0 | 0.7133 | es | 1 Esto a menudo se debe a la hiperventilación, que es un síntoma común en personas con ataques de pánico. 2 Un hormigueo en la nariz puede ser una señal de que está respirando m... |
| 9 | 7549962 | 0 | 0.7019 | es | El entumecimiento y el hormigueo en la cabeza pueden deberse a problemas de salud subyacentes, lesiones, resfriados comunes e incluso ansiedad. Las causas más comunes se enumera... |
| 10 | 7295784 | 0 | 0.6989 | es | Sé dónde están los nervios en la cara y cómo se sienten los síntomas nerviosos. Tengo sensaciones de entumecimiento y hormigueo viajando por ellos. Especialmente el de un lado. |

### qid `1033927`

- query A (`es`): ¿Cuál es el propósito de un modelo conceptual para la práctica avanzada de enfermería?
- query B (`zh`): 高级实践护理概念模型的目的是什么
- diagnosis: TranslationDivergence; nDCG@10 end=100.0000, mix=50.0000, Δ=-50.0000; Recall@10 end=100.0000, mix=100.0000, Δ=0.0000; tokens(a/b)=14/9, len_ratio=1.5556; overlap@10=9; source=evaluate_perquery

Endpoint top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 7211854 | 1 | 0.6811 | es | Los modelos conceptuales y teóricos de enfermería ayudan a aportar conocimientos para mejorar la práctica, orientar la investigación y el currículo e identificar los objetivos d... |
| 2 | 4446553 | 0 | 0.6803 | es | Un modelo conceptual de enfermería desarrollado por Myra Levine. La persona es vista como un ser holístico que se adapta a los desafíos ambientales. En este modelo, el objetivo ... |
| 3 | 7211852 | 0 | 0.6617 | es | Características clave: 1 Se enfoca en aplicar modelos conceptuales en la práctica. 2 Demuestra cómo se aplica una amplia gama de modelos conceptuales de enfermería a la práctica... |
| 4 | 3014088 | 0 | 0.6563 | es | El propósito de este estudio es comparar modelos de cuidados de enfermería funcional y en equipo. parto con resultados de pacientes sensibles a la enfermera en una unidad médico... |
| 5 | 7582710 | 0 | 0.6513 | es | El propósito de este estudio es comparar modelos de cuidados de enfermería funcional y en equipo. parto con resultados de pacientes sensibles a la enfermera en una unidad médico... |
| 6 | 5111824 | 0 | 0.6480 | es | Nuestro modelo de práctica profesional, adaptado de Hoffart y Woods (1996) 1, contiene los valores, las estructuras y los procesos que respaldan el control de las enfermeras reg... |
| 7 | 5393587 | 0 | 0.6463 | es | Una teoría de enfermería, también llamada modelo de enfermería, es un marco desarrollado para guiar a las enfermeras en la forma en que cuidan a sus pacientes. A menudo, estos m... |
| 8 | 1045517 | 0 | 0.6448 | es | Los modelos permiten que los conceptos de la teoría de la enfermería se apliquen con éxito a la práctica de la enfermería. Proporcionan una descripción general del pensamiento d... |
| 9 | 4446557 | 0 | 0.6444 | es | un modelo conceptual de enfermería, formulado por Myra E. levine, preocupado por el mantenimiento de la integridad de la persona. La persona es un ser holístico, un organismo qu... |
| 10 | 873515 | 0 | 0.6435 | es | Información básica sobre el modelo de sinergia de la AACN para la atención al paciente. El concepto central del modelo reconceptualizado de práctica certificada, el Modelo de si... |

Mixed top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 4446553 | 0 | 0.6623 | es | Un modelo conceptual de enfermería desarrollado por Myra Levine. La persona es vista como un ser holístico que se adapta a los desafíos ambientales. En este modelo, el objetivo ... |
| 2 | 3014088 | 0 | 0.6588 | es | El propósito de este estudio es comparar modelos de cuidados de enfermería funcional y en equipo. parto con resultados de pacientes sensibles a la enfermera en una unidad médico... |
| 3 | 7211854 | 1 | 0.6587 | es | Los modelos conceptuales y teóricos de enfermería ayudan a aportar conocimientos para mejorar la práctica, orientar la investigación y el currículo e identificar los objetivos d... |
| 4 | 1045517 | 0 | 0.6581 | zh | 模型使护理理论中的概念能够成功地应用于护理实践。它们概述了理论背后的思想，并可能展示如何将理论引入实践，例如，通过特定的评估方法。 |
| 5 | 873515 | 0 | 0.6511 | es | Información básica sobre el modelo de sinergia de la AACN para la atención al paciente. El concepto central del modelo reconceptualizado de práctica certificada, el Modelo de si... |
| 6 | 7211852 | 0 | 0.6510 | es | Características clave: 1 Se enfoca en aplicar modelos conceptuales en la práctica. 2 Demuestra cómo se aplica una amplia gama de modelos conceptuales de enfermería a la práctica... |
| 7 | 7582710 | 0 | 0.6481 | es | El propósito de este estudio es comparar modelos de cuidados de enfermería funcional y en equipo. parto con resultados de pacientes sensibles a la enfermera en una unidad médico... |
| 8 | 5393587 | 0 | 0.6479 | zh | 护理理论，也称为护理模型，是一个框架，旨在指导护士如何照顾病人。通常，这些框架定义了护理实践，确定了护士的角色，并解释了与护理理论背后的理念相关的护理过程。 |
| 9 | 5111824 | 0 | 0.6465 | es | Nuestro modelo de práctica profesional, adaptado de Hoffart y Woods (1996) 1, contiene los valores, las estructuras y los procesos que respaldan el control de las enfermeras reg... |
| 10 | 4579478 | 0 | 0.6413 | zh | 该项目的广泛目标是通过开发用于评估患者健康/疾病状态和记录整个护理期间的基本信息的手动或计算机辅助工具来改善患者护理。然后将基本信息组织成适合机器处理的护理评估表格.第一章介绍了研究的哲学、目标和范围。第二章考虑了护士的过去、现在、变化和未来的角色。 |

### qid `193422`

- query A (`es`): precio del gas en el monte de galaad oh
- query B (`zh`): 吉列山的汽油价格哦
- diagnosis: RankDrop; nDCG@10 end=100.0000, mix=50.0000, Δ=-50.0000; Recall@10 end=100.0000, mix=100.0000, Δ=0.0000; tokens(a/b)=9/6, len_ratio=1.5000; overlap@10=7; source=evaluate_perquery

Endpoint top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 7962386 | 1 | 0.6406 | es | El ingreso familiar promedio en 43338 (Mount Gilead, OH) de $ 26,845 es aproximadamente un 54 por ciento menos que el ingreso promedio de $ 58,283 para los Estados Unidos en gen... |
| 2 | 7962389 | 0 | 0.6389 | es | Precios de la gasolina en Mount Gilead en YP.com. Vea reseñas, fotos, direcciones, números de teléfono y más para conocer las mejores gasolineras en Mount Gilead, OH. Comience s... |
| 3 | 7962385 | 0 | 0.6249 | es | Busque precios de gasolina baratos en Mount Gilead, Ohio; Encuentre los precios de gasolina y gasolineras locales de Mount Gilead con los mejores precios de combustible. No cone... |
| 4 | 6009056 | 0 | 0.6180 | es | el dia de ayer. $ 2.59 actualizar. Hay 12 informes de precios del gas regular en los últimos 5 días en Hilliard, OH, código postal 43026. El precio promedio del gas regular en H... |
| 5 | 5465936 | 0 | 0.6167 | zh | 周五，Mont Belvieu 的乙烷价格为 22.5 cnts/gal，继续 NGL 的下降趋势，从 2012 年年中开始。上一次我们看到乙烷处于这个水平是在 2002 年。随着天然气价格徘徊在 3.00 美元/MMbtu 以上，毫无疑问。 Mont Belvieu 的塔恩周五公布为 22.5 cnts/gal，继续 NGL从 2012 年年中开始坠... |
| 6 | 3889323 | 0 | 0.6081 | zh | 俄勒冈州天然气价格自 11 月以来首次上涨。根据 AAA 俄勒冈/爱达荷州的数据，现在俄勒冈州普通加仑的平均价格为 1.93 美元。这比一周前增加了三美分。 |
| 7 | 6009058 | 0 | 0.6064 | zh | 过去 5 天内，俄亥俄州希利亚德有 12 份常规天然气价格报告。俄亥俄州希利亚德的平均常规天然气价格为 2.51 美元，比美国全国平均常规天然气价格 2.76 美元低 0.25 美元。最低的常规汽油价格是位于 2567 Walcutt Rd, Hilliard, OH 43026 的 Speedway (#9265) 的 2.42 美元。这里有过去 5... |
| 8 | 2052783 | 0 | 0.5996 | es | 1 gas. 2 El precio de la gasolina en Bend, Oregón es $ 2.30. 3 El precio de la gasolina en Bend es 9.0% más alto que el promedio nacional. 4 Kodiak, Alaska es la ciudad más cara... |
| 9 | 850278 | 0 | 0.5973 | es | Los precios de la gasolina en Yakima subieron con respecto a la misma época del año pasado. PORTLAND, Oregon (AP) - El club automovilístico AAA informa que el precio promedio de... |
| 10 | 2468951 | 0 | 0.5968 | es | en riyadh suadi arabia los precios de la gasolina son increíblemente bajos según un informe reciente del Washington Post El gas subsidiado por el gobierno se vende por solo 45 c... |

Mixed top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 7962389 | 0 | 0.6713 | es | Precios de la gasolina en Mount Gilead en YP.com. Vea reseñas, fotos, direcciones, números de teléfono y más para conocer las mejores gasolineras en Mount Gilead, OH. Comience s... |
| 2 | 7962385 | 0 | 0.6678 | es | Busque precios de gasolina baratos en Mount Gilead, Ohio; Encuentre los precios de gasolina y gasolineras locales de Mount Gilead con los mejores precios de combustible. No cone... |
| 3 | 7962386 | 1 | 0.6625 | es | El ingreso familiar promedio en 43338 (Mount Gilead, OH) de $ 26,845 es aproximadamente un 54 por ciento menos que el ingreso promedio de $ 58,283 para los Estados Unidos en gen... |
| 4 | 6009058 | 0 | 0.6592 | zh | 过去 5 天内，俄亥俄州希利亚德有 12 份常规天然气价格报告。俄亥俄州希利亚德的平均常规天然气价格为 2.51 美元，比美国全国平均常规天然气价格 2.76 美元低 0.25 美元。最低的常规汽油价格是位于 2567 Walcutt Rd, Hilliard, OH 43026 的 Speedway (#9265) 的 2.42 美元。这里有过去 5... |
| 5 | 5465936 | 0 | 0.6506 | zh | 周五，Mont Belvieu 的乙烷价格为 22.5 cnts/gal，继续 NGL 的下降趋势，从 2012 年年中开始。上一次我们看到乙烷处于这个水平是在 2002 年。随着天然气价格徘徊在 3.00 美元/MMbtu 以上，毫无疑问。 Mont Belvieu 的塔恩周五公布为 22.5 cnts/gal，继续 NGL从 2012 年年中开始坠... |
| 6 | 6009056 | 0 | 0.6503 | es | el dia de ayer. $ 2.59 actualizar. Hay 12 informes de precios del gas regular en los últimos 5 días en Hilliard, OH, código postal 43026. El precio promedio del gas regular en H... |
| 7 | 6009057 | 0 | 0.6487 | zh | 2.59 美元更新。过去 5 天内，俄亥俄州希利亚德 (Hilliard) 有 12 份常规天然气价格报告。俄亥俄州希利亚德的平均常规汽油价格为 2.51 美元，比美国全国平均常规汽油价格 2.76 美元低 0.25 美元。最低常规汽油价格为位于 2567 Walcutt Rd, Hilliard, OH 43026 的 Speedway (#9265... |
| 8 | 6854481 | 0 | 0.6466 | zh | 根据 Gasbuddy.com 的数据，美国平均最高和最低的汽油成本在夏威夷为 3.87 美元，在俄克拉荷马州为 2.88 美元。哥斯达黎加的天然气约为每升 650 科朗。所以 3.78 升一加仑等于 2457 科朗，每加仑总计 4.91 美元。大米也因您在该国的位置而异。所有这些价格都与 Playas del Coco/Guanacaste 地区有关... |
| 9 | 2690111 | 0 | 0.6448 | zh | (俄勒冈海岸) ─ ─ ─ ─ AAA 俄勒冈州办公室表示，本周天然气价格将快速上涨，并将在整个夏季保持较高水平。普通无铅汽油的全国平均价格飙升 15 至每加仑 3.64 美元，而俄勒冈州的平均价格在短短一周内上涨了 1 角钱至 3.86 美元。年度最高。 “全国平均价格现在已经连续 9 天上涨，但仍比今年 2 月 27 日的 3.79 美元的峰值价格... |
| 10 | 850278 | 0 | 0.6442 | es | Los precios de la gasolina en Yakima subieron con respecto a la misma época del año pasado. PORTLAND, Oregon (AP) - El club automovilístico AAA informa que el precio promedio de... |

### qid `408945`

- query A (`es`): ¿Dylan O'Brien es pariente de Adam Brody?
- query B (`zh`): 迪伦·奥布莱恩和亚当·布罗迪有关系吗
- diagnosis: RankDrop; nDCG@10 end=100.0000, mix=50.0000, Δ=-50.0000; Recall@10 end=100.0000, mix=100.0000, Δ=0.0000; tokens(a/b)=8/8, len_ratio=1.0000; overlap@10=9; source=evaluate_perquery

Endpoint top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 7692287 | 1 | 0.6012 | es | La fama de Dylan O'Brien se interpone en sus carreras de Chipotle. Y es cierto que, como el mejor amigo del canino angustiado titular del programa, O'Brien, de 23 años, lleva lo... |
| 2 | 2849057 | 0 | 0.5939 | es | Dylan O'Brien Actor, The Maze Runner Dylan O'Brien nació en la ciudad de Nueva York, hijo de Lisa Rhodes, una ex actriz que también dirigía una escuela de actuación, y Patrick B... |
| 3 | 1090490 | 0 | 0.5901 | zh | 概述（2）。 Dylan O'Brien 出生于纽约市，前女演员 Lisa Rhodes 和一名摄影师帕特里克 B. O'Brien 出生，他的父亲是爱尔兰血统，母亲是英国人，西班牙和意大利血统。迪伦在新泽西州联合县的斯普林菲尔德镇长大，12 岁时随家人搬到加利福尼亚的赫莫萨海滩。ini Bio (1)。迪伦·奥布莱恩 (Dylan O'Brien) ... |
| 4 | 1090493 | 0 | 0.5881 | zh | 官方照片Ãƒâ€šÃ‚Â»。 Dylan O'Brien 出生于纽约市，前女演员 Lisa Rhodes 和一名摄影师帕特里克 B. O'Brien 出生，他的父亲是爱尔兰血统，母亲是英国人，西班牙和意大利的祖先。官方照片Ãƒâ€šÃ‚Â»。迪伦·奥布莱恩 (Dylan O'Brien) 出生于纽约市，父亲是前女演员丽莎·罗德斯 (Lisa Rhode... |
| 5 | 1090492 | 0 | 0.5879 | zh | 演员。官方照片Ãƒâ€šÃ‚Â»。迪伦·奥布莱恩 (Dylan O'Brien) 出生于纽约市，父亲是前女演员丽莎·罗德斯 (Lisa Rhodes) 和摄影师帕特里克·B·奥布莱恩 (Patrick B. O'Brien)，后者还经营着一所表演学校。他的父亲有爱尔兰血统，他的母亲有英国、西班牙和意大利血统。官方照片â€šâ€。迪伦·奥布莱恩 (Dyl... |
| 6 | 4215800 | 0 | 0.5771 | zh | 迪伦·奥布莱恩 (Dylan O'Brien) 于 1990 年代初出生于美国纽约州纽约市，是丽莎·奥布莱恩 (Lisa O'Brien) 和帕特里克·奥布莱恩 (Patrick O'Brien) 的一位才华横溢的演员，以其愚蠢的个性而闻名。他在新泽西州斯普林菲尔德镇长大。他有爱尔兰、意大利、英国和西班牙血统。 |
| 7 | 3517281 | 0 | 0.5768 | zh | Dylan O'Brien (II) Dylan O'Brien 出生于纽约市，她的父亲是前女演员丽莎·罗德斯 (Lisa Rhodes) 和一名摄影师帕特里克·奥布莱恩 (Patrick B. O'Brien)。他的父亲有爱尔兰血统，母亲有英国、西班牙和意大利血统。 |
| 8 | 4215807 | 0 | 0.5742 | zh | 个人生活 迪伦·奥布莱恩 (Dylan O'Brien) 的父亲是一位前女演员，他还经营着一所表演学校，而帕特里克·奥布莱恩 (Patrick O'Brien) 则是一名摄影师。自从他们在 2011 年第一次见面以来，他就与女演员布丽特·罗伯逊建立了关系。 |
| 9 | 7692283 | 0 | 0.5695 | es | * 12,5% italiano. * 12,5% españoles. Dylan OÃƒÂ ¢ Ã‚â‚¬Ã‚â „¢ Brien es un actor, músico y director estadounidense. El padre de DylanÃƒÂ ¢ Ã‚â‚¬Ã‚â „¢ es de ascendencia irlandesa... |
| 10 | 2503162 | 0 | 0.5695 | zh | 迪伦·奥布莱恩。迪伦·奥布莱恩 (Dylan O'Brien) 出生于纽约市，父亲是前女演员丽莎·罗德斯 (Lisa Rhodes) 和摄影师帕特里克·B·奥布莱恩 (Patrick B. O'Brien)，后者还经营着一所表演学校。他的父亲有爱尔兰血统，母亲有英国、西班牙和意大利血统。 |

Mixed top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 4215807 | 0 | 0.6169 | zh | 个人生活 迪伦·奥布莱恩 (Dylan O'Brien) 的父亲是一位前女演员，他还经营着一所表演学校，而帕特里克·奥布莱恩 (Patrick O'Brien) 则是一名摄影师。自从他们在 2011 年第一次见面以来，他就与女演员布丽特·罗伯逊建立了关系。 |
| 2 | 1090492 | 0 | 0.6132 | zh | 演员。官方照片Ãƒâ€šÃ‚Â»。迪伦·奥布莱恩 (Dylan O'Brien) 出生于纽约市，父亲是前女演员丽莎·罗德斯 (Lisa Rhodes) 和摄影师帕特里克·B·奥布莱恩 (Patrick B. O'Brien)，后者还经营着一所表演学校。他的父亲有爱尔兰血统，他的母亲有英国、西班牙和意大利血统。官方照片â€šâ€。迪伦·奥布莱恩 (Dyl... |
| 3 | 7692287 | 1 | 0.6127 | es | La fama de Dylan O'Brien se interpone en sus carreras de Chipotle. Y es cierto que, como el mejor amigo del canino angustiado titular del programa, O'Brien, de 23 años, lleva lo... |
| 4 | 1090493 | 0 | 0.6084 | zh | 官方照片Ãƒâ€šÃ‚Â»。 Dylan O'Brien 出生于纽约市，前女演员 Lisa Rhodes 和一名摄影师帕特里克 B. O'Brien 出生，他的父亲是爱尔兰血统，母亲是英国人，西班牙和意大利的祖先。官方照片Ãƒâ€šÃ‚Â»。迪伦·奥布莱恩 (Dylan O'Brien) 出生于纽约市，父亲是前女演员丽莎·罗德斯 (Lisa Rhode... |
| 5 | 1090490 | 0 | 0.6084 | zh | 概述（2）。 Dylan O'Brien 出生于纽约市，前女演员 Lisa Rhodes 和一名摄影师帕特里克 B. O'Brien 出生，他的父亲是爱尔兰血统，母亲是英国人，西班牙和意大利血统。迪伦在新泽西州联合县的斯普林菲尔德镇长大，12 岁时随家人搬到加利福尼亚的赫莫萨海滩。ini Bio (1)。迪伦·奥布莱恩 (Dylan O'Brien) ... |
| 6 | 4215800 | 0 | 0.6050 | zh | 迪伦·奥布莱恩 (Dylan O'Brien) 于 1990 年代初出生于美国纽约州纽约市，是丽莎·奥布莱恩 (Lisa O'Brien) 和帕特里克·奥布莱恩 (Patrick O'Brien) 的一位才华横溢的演员，以其愚蠢的个性而闻名。他在新泽西州斯普林菲尔德镇长大。他有爱尔兰、意大利、英国和西班牙血统。 |
| 7 | 2503162 | 0 | 0.5998 | zh | 迪伦·奥布莱恩。迪伦·奥布莱恩 (Dylan O'Brien) 出生于纽约市，父亲是前女演员丽莎·罗德斯 (Lisa Rhodes) 和摄影师帕特里克·B·奥布莱恩 (Patrick B. O'Brien)，后者还经营着一所表演学校。他的父亲有爱尔兰血统，母亲有英国、西班牙和意大利血统。 |
| 8 | 3517281 | 0 | 0.5823 | zh | Dylan O'Brien (II) Dylan O'Brien 出生于纽约市，她的父亲是前女演员丽莎·罗德斯 (Lisa Rhodes) 和一名摄影师帕特里克·奥布莱恩 (Patrick B. O'Brien)。他的父亲有爱尔兰血统，母亲有英国、西班牙和意大利血统。 |
| 9 | 2849057 | 0 | 0.5810 | zh | Dylan O'Brien 演员，The Maze Runner Dylan O'Brien 出生于纽约市，她的父亲是 Lisa Rhodes，Lisa Rhodes 是一位前女演员，同时还经营着一所表演学校，以及摄影师 Patrick B. O'Brien。他的父亲有爱尔兰血统，他的母亲有英国、西班牙和意大利血统。在《皇家赌场》、《安慰量子》和《天幕... |
| 10 | 4215804 | 0 | 0.5797 | zh | Mini Bio (1) Dylan O'Brien 出生于纽约市，她的父亲是前女演员 Lisa Rhodes，她还经营着一所表演学校，以及摄影师 Patrick B. O'Brien。他的父亲有爱尔兰血统，母亲有英国、西班牙和意大利血统。 |

### qid `414733`

- query A (`es`): ¿Es la ley en California que su hijo tiene que ser vacunado?
- query B (`zh`): 加州法律规定您的孩子必须接种疫苗吗
- diagnosis: RankDrop; nDCG@10 end=100.0000, mix=50.0000, Δ=-50.0000; Recall@10 end=100.0000, mix=100.0000, Δ=0.0000; tokens(a/b)=12/10, len_ratio=1.2000; overlap@10=10; source=evaluate_perquery

Endpoint top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 7444061 | 1 | 0.7272 | es | Ley que exige que los padres de California vacunen a sus hijos que probablemente aprueben. (CBS SF) ÃƒÂ ¢ Ã‚â‚¬Ã‚â € El jueves pasado, se introdujo una ley estatal que, si se ap... |
| 2 | 3745290 | 0 | 0.7223 | es | 1 La ley de California requiere que todos los niños inscritos en escuelas públicas, tanto públicas como privadas, tengan ciertas vacunas recomendadas por los médicos o las recib... |
| 3 | 1259666 | 0 | 0.7194 | es | La ley de California requiere que los niños estén vacunados. Los niños están exentos de los requisitos de vacunación solo si un padre o tutor presenta una declaración por escrit... |
| 4 | 7444062 | 0 | 0.7181 | es | Kristen Kinne fotografiada con su hija en Calero Park cerca de su casa en San José, California, el miércoles 29 de junio de 2016. La controvertida ley de California que obliga a... |
| 5 | 4676081 | 0 | 0.7147 | es | El sobreviviente de leucemia Rhett Krawitt, de 7 años, habla con el senador Ben Allen, demócrata de Santa Mónica, quien es el coautor de una medida que requiere que casi todos l... |
| 6 | 8622255 | 0 | 0.7137 | es | La ley de California requiere que todos los niños inscritos en escuelas públicas, tanto públicas como privadas, tengan ciertas vacunas recomendadas por los médicos o las reciban... |
| 7 | 7444068 | 0 | 0.7120 | es | El 30 de junio, el gobernador de California, Jerry Brown, promulgó una ley que requiere la vacunación de todos los niños en la escuela o guardería, excepto por las exenciones qu... |
| 8 | 7444065 | 0 | 0.7046 | es | Comparta esto: El senador estatal Ben Allen, demócrata por Santa Mónica, a la derecha, y el senador Richard Pan, demócrata por Sacramento, hablan con los medios después de que s... |
| 9 | 697485 | 0 | 0.6967 | es | La nueva y estricta ley de vacunación de California es legalmente sólida y servirá como modelo sobre cómo mantener saludables a los niños, dicen los profesores de Stanford. El 2... |
| 10 | 4676086 | 0 | 0.6958 | es | Gavin Wonnacott le señala un letrero a su madre, Jennifer, en Sacramento, donde el gobernador Jerry Brown firmó una legislación polémica que exige que casi todos los niños en ed... |

Mixed top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 1259666 | 0 | 0.7363 | zh | 加州法律要求儿童接种疫苗。仅当父母或监护人提交由执业医师（MD 或 DO）出具的书面声明声明： 儿童的身体状况或医疗情况导致未表明需要进行免疫接种时，儿童才可以免除免疫接种要求. |
| 2 | 3745290 | 0 | 0.7363 | es | 1 La ley de California requiere que todos los niños inscritos en escuelas públicas, tanto públicas como privadas, tengan ciertas vacunas recomendadas por los médicos o las recib... |
| 3 | 7444061 | 1 | 0.7344 | es | Ley que exige que los padres de California vacunen a sus hijos que probablemente aprueben. (CBS SF) ÃƒÂ ¢ Ã‚â‚¬Ã‚â € El jueves pasado, se introdujo una ley estatal que, si se ap... |
| 4 | 8622255 | 0 | 0.7328 | zh | 加州法律要求在公立和私立公立学校就读的所有儿童都必须接受医生推荐的某些免疫接种，或者在入学时接受这些疫苗。 |
| 5 | 4676081 | 0 | 0.7324 | es | El sobreviviente de leucemia Rhett Krawitt, de 7 años, habla con el senador Ben Allen, demócrata de Santa Mónica, quien es el coautor de una medida que requiere que casi todos l... |
| 6 | 7444068 | 0 | 0.7247 | es | El 30 de junio, el gobernador de California, Jerry Brown, promulgó una ley que requiere la vacunación de todos los niños en la escuela o guardería, excepto por las exenciones qu... |
| 7 | 7444062 | 0 | 0.7193 | es | Kristen Kinne fotografiada con su hija en Calero Park cerca de su casa en San José, California, el miércoles 29 de junio de 2016. La controvertida ley de California que obliga a... |
| 8 | 7444065 | 0 | 0.7087 | es | Comparta esto: El senador estatal Ben Allen, demócrata por Santa Mónica, a la derecha, y el senador Richard Pan, demócrata por Sacramento, hablan con los medios después de que s... |
| 9 | 4676086 | 0 | 0.7074 | es | Gavin Wonnacott le señala un letrero a su madre, Jennifer, en Sacramento, donde el gobernador Jerry Brown firmó una legislación polémica que exige que casi todos los niños en ed... |
| 10 | 697485 | 0 | 0.7038 | es | La nueva y estricta ley de vacunación de California es legalmente sólida y servirá como modelo sobre cómo mantener saludables a los niños, dicen los profesores de Stanford. El 2... |

### qid `619805`

- query A (`es`): ¿Qué obtuvieron los ganadores del maratón de boston 2016?
- query B (`zh`): 2016 年波士顿马拉松赛获胜者得到了什么
- diagnosis: RankDrop; nDCG@10 end=100.0000, mix=50.0000, Δ=-50.0000; Recall@10 end=100.0000, mix=100.0000, Δ=0.0000; tokens(a/b)=8/9, len_ratio=0.8889; overlap@10=8; source=evaluate_perquery

Endpoint top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 7650075 | 1 | 0.6347 | es | Aquí, intentaré averiguar cuánto dinero están ganando. Lo mejor que puedo decir es que hay cuatro formas en que los mejores estadounidenses pueden sacar provecho de una gran vic... |
| 2 | 3108095 | 0 | 0.6327 | es | aquí s quienes ganaron sus respectivas carreras en la 119a edición del maratón de boston por kyle clauss boston daily 20 de abril de 2015 11 04 am foto vía ap men s elite lelisa... |
| 3 | 2904260 | 0 | 0.6324 | zh | 慈善机构也赢了：仅去年一年，跑步者为 300 多项事业筹集了创纪录的 3840 万美元。当地经济也是如此：组织比赛的波士顿体育协会估计将产生约 1.82 亿美元的收入。琐事。用这个马拉松细节给您的朋友留下深刻印象或赢得游戏节目。 |
| 4 | 2286425 | 0 | 0.6306 | es | Maratón de Boston 2016. Más de 30.000 participantes se registraron para el Maratón de Boston 2016, el tercer campo más grande en la historia de la carrera. Los corredores duerme... |
| 5 | 2502622 | 0 | 0.6301 | es | Análisis del maratón de Boston 2016. Se acerca el día de la carrera del maratón de Boston; y se vuelve aún más emocionante cuando se asignan los números de dorsal. BAA acaba de ... |
| 6 | 2904268 | 0 | 0.6283 | es | Comentarios. BOSTON (AP) - El maratón de Boston incluye 42 km, 34 atletas de élite, 30.000 corredores, 87 países, 1 millón de espectadores y 830.500 dólares en premios. |
| 7 | 3108093 | 0 | 0.6257 | es | crédito de imagen charles krupa ap foto el hombre estadounidense que ganó el maratón de boston hoy honró a las víctimas del bombardeo del año pasado al tener sus nombres escrito... |
| 8 | 553498 | 0 | 0.6197 | es | Desglose del dinero del premio del Maratón de Boston. Lo crea o no, puede ganar mucho dinero colocándose en el Maratón de Boston, pero ¿cuánto dinero obtienen los primeros clasi... |
| 9 | 7812736 | 0 | 0.6149 | es | En 1975, Boston se convirtió en el primer maratón importante en incluir una división de sillas de ruedas. En 1986, se otorgó un premio en metálico a los ganadores del maratón de... |
| 10 | 2257564 | 0 | 0.6096 | es | Yemane Adhane Tsegay, otro etíope, estaba otros 30 segundos atrás en el tercer lugar. Hayle, de 21 años, es el ganador más joven de Boston desde que Shigeki Tanaka ganó a los 19... |

Mixed top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 2904260 | 0 | 0.6673 | zh | 慈善机构也赢了：仅去年一年，跑步者为 300 多项事业筹集了创纪录的 3840 万美元。当地经济也是如此：组织比赛的波士顿体育协会估计将产生约 1.82 亿美元的收入。琐事。用这个马拉松细节给您的朋友留下深刻印象或赢得游戏节目。 |
| 2 | 2904268 | 0 | 0.6614 | zh | 注释。波士顿（美联社）——波士顿马拉松赛包括 26.2 英里、34 名精英运动员、30,000 名跑步者、87 个国家、100 万名观众和 830,500 美元的奖金。 |
| 3 | 7650075 | 1 | 0.6415 | es | Aquí, intentaré averiguar cuánto dinero están ganando. Lo mejor que puedo decir es que hay cuatro formas en que los mejores estadounidenses pueden sacar provecho de una gran vic... |
| 4 | 2502622 | 0 | 0.6312 | es | Análisis del maratón de Boston 2016. Se acerca el día de la carrera del maratón de Boston; y se vuelve aún más emocionante cuando se asignan los números de dorsal. BAA acaba de ... |
| 5 | 3527470 | 0 | 0.6309 | zh | 第一波参赛者于上午 8:50 开始，精英跑者在上午 9:30 刚过就出发。在这里观看波士顿马拉松直播。如果您需要更多信息，我们也有。无论您有爱国者日免费参加 2016 年波士顿马拉松比赛，还是希望在家或办公桌上观看比赛，这里都有您在比赛日所需的电视、直播和日程安排信息。您可以在多个站点、多个波士顿电视台附属机构或 WBZ 1030 AM 的广播上实时观... |
| 6 | 553498 | 0 | 0.6301 | zh | 波士顿马拉松奖金明细。信不信由你，你可以通过参加波士顿马拉松赢得大笔奖金，但顶级选手能拿到多少钱？波士顿马拉松今天是第 121 次在爱国者日举行。对于波士顿人来说，这一直是一个特殊的日子，在 2012 年恐怖袭击震撼终点线之后，这一天变得更加强大。从那时起，马拉松对波士顿人来说有了新的意义。不仅是波士顿人，还有全国各地的美国人。 |
| 7 | 7650077 | 0 | 0.6292 | zh | 继续阅读主要故事。波士顿 - 在 Lelisa Desisa 在 2013 年赢得波士顿马拉松比赛两个小时后，两枚炸弹穿过终点区，炸死三名观众，炸伤 264 人。事后，Desisa 与波士顿人民建立了联系，并将他的获胜者奖牌赠送给了这座城市。OSTON â€â€ 两小时后Lelisa Desisa 在 2013 年赢得波士顿马拉松赛冠军，两枚炸弹穿过终... |
| 8 | 2286425 | 0 | 0.6279 | es | Maratón de Boston 2016. Más de 30.000 participantes se registraron para el Maratón de Boston 2016, el tercer campo más grande en la historia de la carrera. Los corredores duerme... |
| 9 | 3108093 | 0 | 0.6208 | es | crédito de imagen charles krupa ap foto el hombre estadounidense que ganó el maratón de boston hoy honró a las víctimas del bombardeo del año pasado al tener sus nombres escrito... |
| 10 | 7812736 | 0 | 0.6196 | zh | 1975 年，波士顿成为第一个包含轮椅组的主要马拉松赛事。 1986 年，奖金首次颁发给波士顿马拉松赛的获胜者。约翰·A·凯利 (John A. Kelley) 于 1928 年参加了他的第一场波士顿马拉松比赛，保持着比赛开始次数最多 (61) 和完成次数最多 (58) 的记录。1986 年，奖金首次颁发给波士顿马拉松赛的获胜者。约翰·A·凯利 (Jo... |

### qid `809798`

- query A (`es`): ¿Cuál es el nombre del mayordomo en la familia Addams?
- query B (`zh`): 亚当斯家族的管家叫什么名字
- diagnosis: RankDrop; nDCG@10 end=100.0000, mix=50.0000, Δ=-50.0000; Recall@10 end=100.0000, mix=100.0000, Δ=0.0000; tokens(a/b)=10/7, len_ratio=1.4286; overlap@10=7; source=evaluate_perquery

Endpoint top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 7803851 | 1 | 0.6258 | es | Lurch (La familia Addams) - Caracterización. 1 Lurch es un mayordomo sombrío, tembloroso y de 6 pies 9 pulgadas (2,05 m) de altura que se parece un poco a un cruce entre el mons... |
| 2 | 2974482 | 0 | 0.6102 | es | La familia Addams. Para otros usos, consulte La familia Addams (desambiguación). La familia Addams es un hogar ficticio creado por el dibujante estadounidense Charles Addams. Lo... |
| 3 | 4942753 | 0 | 0.5932 | es | Pie de imagen La familia Addams presentaba una familia de bichos raros macabros, incluido Pugsley (izquierda). Ken Weatherwax, quien interpretó a Pugsley en la familia Addams or... |
| 4 | 2974484 | 0 | 0.5926 | es | De izquierda a derecha, Pugsley, Wednesday, Gómez, Aristóteles el pulpo, Tío Fester, Morticia. La familia Addams es un hogar ficticio creado por el dibujante estadounidense Char... |
| 5 | 4876625 | 0 | 0.5921 | es | Alan Napier nació de una prestigiosa raza. Era primo del primer ministro británico Neville Chamberlain y descendiente directo de Charles Dickens. Aunque su carrera como actor ..... |
| 6 | 2276149 | 0 | 0.5879 | es | La familia Addams, de Charles Addams. La familia Addams es un grupo de personajes de ficción creados por el dibujante estadounidense Charles Addams. Los personajes de la familia... |
| 7 | 4942756 | 0 | 0.5862 | es | Pugsley Addams es miembro de la familia ficticia Addams, creada por el dibujante estadounidense Charles Addams. La serie de 1998 The New Addams Family presenta a Pugsley interpr... |
| 8 | 7803848 | 0 | 0.5855 | es | La familia Addams, de Charles Addams. La familia Addams es un grupo de personajes de ficción creados por el dibujante estadounidense Charles Addams. Los personajes de la familia... |
| 9 | 1313288 | 0 | 0.5843 | es | Ted Cassidy (1932ÃƒÂ ¢ Ã‚â‚¬Ã‚â € œ1979) Ted Cassidy nació en Pittsburgh, Pennsylvania y se crió en Philippi, West Virginia. Fue un actor muy respetado que interpretó a muchos p... |
| 10 | 4491271 | 0 | 0.5788 | es | Alan Napier nació de una prestigiosa raza. Era primo del primer ministro británico Neville Chamberlain y descendiente directo de Charles Dickens. Aunque su carrera como actor ..... |

Mixed top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 2974482 | 0 | 0.6280 | es | La familia Addams. Para otros usos, consulte La familia Addams (desambiguación). La familia Addams es un hogar ficticio creado por el dibujante estadounidense Charles Addams. Lo... |
| 2 | 7803846 | 0 | 0.6183 | zh | Lurch (The Addams Family) Lurch（名字不详）是美国漫画家查尔斯·亚当斯创作的虚构人物，作为亚当斯家族的男仆。在最初的电视剧中，Lurch 由 Ted Cassidy 扮演，他使用了著名的标语，你响了？ |
| 3 | 7803851 | 1 | 0.6143 | es | Lurch (La familia Addams) - Caracterización. 1 Lurch es un mayordomo sombrío, tembloroso y de 6 pies 9 pulgadas (2,05 m) de altura que se parece un poco a un cruce entre el mons... |
| 4 | 2974484 | 0 | 0.6109 | zh | 从左到右，Pugsley，星期三，Gomez，章鱼亚里士多德，Fester 叔叔，Morticia。亚当斯一家是美国漫画家查尔斯·亚当斯创作的虚构家庭。亚当斯家族的角色传统上包括戈麦斯、莫蒂西亚、费斯特叔叔、鲁奇、祖母、星期三、帕格斯利和事物。 |
| 5 | 3863729 | 0 | 0.6101 | zh | 了解住在和平场老房子里的亚当斯家族的四代人。 |
| 6 | 2276149 | 0 | 0.6061 | es | La familia Addams, de Charles Addams. La familia Addams es un grupo de personajes de ficción creados por el dibujante estadounidense Charles Addams. Los personajes de la familia... |
| 7 | 2780984 | 0 | 0.5993 | zh | ADDAMS FAMILY 以原创故事为特色，这是每个父亲的噩梦。星期三亚当斯，黑暗的终极公主，已经长大并爱上了一个来自一个受人尊敬的家庭的可爱聪明的年轻人——一个她父母从未见过的男人。他的音乐喜剧，由大卫·布莱恩特 (David Bryant) 执导，收录了查尔斯·亚当斯 (Charles Addams) 在《纽约客》(The New Yorker)... |
| 8 | 7803848 | 0 | 0.5989 | es | La familia Addams, de Charles Addams. La familia Addams es un grupo de personajes de ficción creados por el dibujante estadounidense Charles Addams. Los personajes de la familia... |
| 9 | 4876625 | 0 | 0.5976 | es | Alan Napier nació de una prestigiosa raza. Era primo del primer ministro británico Neville Chamberlain y descendiente directo de Charles Dickens. Aunque su carrera como actor ..... |
| 10 | 4942756 | 0 | 0.5962 | es | Pugsley Addams es miembro de la familia ficticia Addams, creada por el dibujante estadounidense Charles Addams. La serie de 1998 The New Addams Family presenta a Pugsley interpr... |

### qid `337190`

- query A (`es`): cuantos años para beber en iowa
- query B (`zh`): 在爱荷华州喝酒多少岁
- diagnosis: RecallDrop; nDCG@10 end=43.0677, mix=0.0000, Δ=-43.0677; Recall@10 end=100.0000, mix=0.0000, Δ=-100.0000; tokens(a/b)=6/7, len_ratio=0.8571; overlap@10=8; source=evaluate_perquery

Endpoint top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 4044895 | 0 | 0.7139 | zh | 最佳答案：俄亥俄州和美国其他所有州的法定饮酒年龄很长一段时间都是 21 岁。时间。 |
| 2 | 5280998 | 0 | 0.6932 | zh | 华盛顿特区：啤酒和葡萄酒的法定饮酒年龄是 18 岁，白酒是 21 岁。伊利诺伊州：啤酒和葡萄酒的法定饮酒年龄是 19 岁，白酒是 21 岁。堪萨斯州：3.2 ABV 啤酒的法定饮酒年龄为 18 岁，3.2 ABV 以上的啤酒、葡萄酒和白酒的法定饮酒年龄为 21 岁。 |
| 3 | 2779899 | 0 | 0.6895 | zh | 两个年龄限制均适用于以下州： 华盛顿特区：啤酒和葡萄酒的法定饮酒年龄为 18 岁，酒为 21 岁。伊利诺伊州：啤酒和葡萄酒的法定饮酒年龄是 19 岁，白酒是 21 岁。堪萨斯州：3.2 ABV 啤酒的法定饮酒年龄为 18 岁，3.2 ABV 以上的啤酒、葡萄酒和白酒的法定饮酒年龄为 21 岁。马里兰州：啤酒和葡萄酒的法定饮酒年龄是 18 岁，白酒是 2... |
| 4 | 7386277 | 1 | 0.6857 | es | Leyes de alcohol de Iowa Dónde comprar alcohol Las tiendas de paquetes de propiedad estatal, a veces llamadas tiendas ABC, son el único lugar para comprar licor fuerte en Iowa. ... |
| 5 | 4689812 | 0 | 0.6845 | zh | 华盛顿特区：啤酒和葡萄酒的法定饮酒年龄是 18 岁，白酒是 21 岁。伊利诺伊州：啤酒和葡萄酒的法定饮酒年龄是 19 岁，白酒是 21 岁。堪萨斯州：3.2 ABV 啤酒的法定饮酒年龄为 18 岁，3.2 ABV 以上的啤酒、葡萄酒和白酒的法定饮酒年龄为 21 岁。马里兰州：啤酒和葡萄酒的法定饮酒年龄是 18 岁，白酒是 21 岁。 |
| 6 | 2684426 | 0 | 0.6840 | zh | 华盛顿特区：啤酒和葡萄酒的法定饮酒年龄为 18 岁，白酒为 21 岁。伊利诺伊州：啤酒和葡萄酒的法定饮酒年龄是 19 岁，白酒是 21 岁。堪萨斯州：3.2 ABV 啤酒的法定饮酒年龄为 18 岁，3.2 ABV 以上的啤酒、葡萄酒和白酒的法定饮酒年龄为 21 岁。马里兰州：啤酒和葡萄酒的法定饮酒年龄是 18 岁，白酒是 21 岁。 |
| 7 | 7785764 | 0 | 0.6837 | zh | 华盛顿特区：啤酒和葡萄酒的法定饮酒年龄是 18 岁，白酒是 21 岁。伊利诺伊州：啤酒和葡萄酒的法定饮酒年龄是 19 岁，白酒是 21 岁。堪萨斯州：3.2 ABV 啤酒的法定饮酒年龄为 18 岁，3.2 ABV 以上的啤酒、葡萄酒和白酒的法定饮酒年龄为 21 岁。马里兰州：啤酒和葡萄酒的法定饮酒年龄是 18 岁，白酒是 21 岁。北卡罗来纳州：啤酒和... |
| 8 | 7044752 | 0 | 0.6833 | zh | 华盛顿特区：啤酒和葡萄酒的法定饮酒年龄是 18 岁，白酒是 21 岁。伊利诺伊州：啤酒和葡萄酒的法定饮酒年龄是 19 岁，白酒是 21 岁。堪萨斯州：3.2 ABV 啤酒的法定饮酒年龄为 18 岁，3.2 ABV 以上的啤酒、葡萄酒和白酒的法定饮酒年龄为 21 岁。马里兰州：啤酒和葡萄酒的法定饮酒年龄是 18 岁，白酒是 21 岁。北卡罗来纳州：啤酒和... |
| 9 | 5887300 | 0 | 0.6779 | zh | 快速回答。根据伊利诺伊州酒类控制委员会的规定，截至 2015 年 1 月，一个人必须年满 18 岁才能在伊利诺伊州提供酒精饮料。但是，伊利诺伊州酒类控制法允许对该问题进行司法控制，伊利诺伊州的一些地方要求年龄超过 18 岁。继续阅读。 |
| 10 | 1365804 | 0 | 0.6767 | es | La cerveza y el vino se venden en tiendas minoristas. Ambos tipos de tiendas pueden vender alcohol de 6 a.m. a 2 a.m. de lunes a sábado y de 8 a.m. a 2 a.m. los domingos. Edad l... |

Mixed top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 4044895 | 0 | 0.7600 | zh | 最佳答案：俄亥俄州和美国其他所有州的法定饮酒年龄很长一段时间都是 21 岁。时间。 |
| 2 | 5280998 | 0 | 0.7421 | zh | 华盛顿特区：啤酒和葡萄酒的法定饮酒年龄是 18 岁，白酒是 21 岁。伊利诺伊州：啤酒和葡萄酒的法定饮酒年龄是 19 岁，白酒是 21 岁。堪萨斯州：3.2 ABV 啤酒的法定饮酒年龄为 18 岁，3.2 ABV 以上的啤酒、葡萄酒和白酒的法定饮酒年龄为 21 岁。 |
| 3 | 2779899 | 0 | 0.7396 | zh | 两个年龄限制均适用于以下州： 华盛顿特区：啤酒和葡萄酒的法定饮酒年龄为 18 岁，酒为 21 岁。伊利诺伊州：啤酒和葡萄酒的法定饮酒年龄是 19 岁，白酒是 21 岁。堪萨斯州：3.2 ABV 啤酒的法定饮酒年龄为 18 岁，3.2 ABV 以上的啤酒、葡萄酒和白酒的法定饮酒年龄为 21 岁。马里兰州：啤酒和葡萄酒的法定饮酒年龄是 18 岁，白酒是 2... |
| 4 | 5887300 | 0 | 0.7355 | zh | 快速回答。根据伊利诺伊州酒类控制委员会的规定，截至 2015 年 1 月，一个人必须年满 18 岁才能在伊利诺伊州提供酒精饮料。但是，伊利诺伊州酒类控制法允许对该问题进行司法控制，伊利诺伊州的一些地方要求年龄超过 18 岁。继续阅读。 |
| 5 | 4689812 | 0 | 0.7339 | zh | 华盛顿特区：啤酒和葡萄酒的法定饮酒年龄是 18 岁，白酒是 21 岁。伊利诺伊州：啤酒和葡萄酒的法定饮酒年龄是 19 岁，白酒是 21 岁。堪萨斯州：3.2 ABV 啤酒的法定饮酒年龄为 18 岁，3.2 ABV 以上的啤酒、葡萄酒和白酒的法定饮酒年龄为 21 岁。马里兰州：啤酒和葡萄酒的法定饮酒年龄是 18 岁，白酒是 21 岁。 |
| 6 | 2684426 | 0 | 0.7338 | zh | 华盛顿特区：啤酒和葡萄酒的法定饮酒年龄为 18 岁，白酒为 21 岁。伊利诺伊州：啤酒和葡萄酒的法定饮酒年龄是 19 岁，白酒是 21 岁。堪萨斯州：3.2 ABV 啤酒的法定饮酒年龄为 18 岁，3.2 ABV 以上的啤酒、葡萄酒和白酒的法定饮酒年龄为 21 岁。马里兰州：啤酒和葡萄酒的法定饮酒年龄是 18 岁，白酒是 21 岁。 |
| 7 | 4044899 | 0 | 0.7322 | zh | 爱达荷州的饮酒年龄在 1972 年降低到 19 岁。1987 年 4 月提高到 21 岁，但有一个祖父条款，允许那些在 c 时年满 19 岁或 20 岁的人¦ 允许在 21 岁之前购买酒精。1982 年 8 月 18 日下午 1:59 从 18 变为 19，然后 1983 年 8 月 18 日晚上 11:59 从 19 变为 21。我是 1964 年 ... |
| 8 | 7785764 | 0 | 0.7313 | zh | 华盛顿特区：啤酒和葡萄酒的法定饮酒年龄是 18 岁，白酒是 21 岁。伊利诺伊州：啤酒和葡萄酒的法定饮酒年龄是 19 岁，白酒是 21 岁。堪萨斯州：3.2 ABV 啤酒的法定饮酒年龄为 18 岁，3.2 ABV 以上的啤酒、葡萄酒和白酒的法定饮酒年龄为 21 岁。马里兰州：啤酒和葡萄酒的法定饮酒年龄是 18 岁，白酒是 21 岁。北卡罗来纳州：啤酒和... |
| 9 | 7044752 | 0 | 0.7296 | zh | 华盛顿特区：啤酒和葡萄酒的法定饮酒年龄是 18 岁，白酒是 21 岁。伊利诺伊州：啤酒和葡萄酒的法定饮酒年龄是 19 岁，白酒是 21 岁。堪萨斯州：3.2 ABV 啤酒的法定饮酒年龄为 18 岁，3.2 ABV 以上的啤酒、葡萄酒和白酒的法定饮酒年龄为 21 岁。马里兰州：啤酒和葡萄酒的法定饮酒年龄是 18 岁，白酒是 21 岁。北卡罗来纳州：啤酒和... |
| 10 | 7386276 | 0 | 0.7257 | zh | ：啤酒和葡萄酒的法定饮酒年龄是 18 岁，白酒是 21 岁。伊利诺伊州：啤酒和葡萄酒的法定饮酒年龄是 19 岁，白酒是 21 岁。堪萨斯州：3.2% ABW 啤酒的法定饮酒年龄为 18 岁，而 ABW 3.2% 以上的啤酒、葡萄酒和白酒的法定饮酒年龄为 21 岁。马里兰州：啤酒和葡萄酒的法定饮酒年龄是 18 岁，白酒是 21 岁。 |

### qid `1005949`

- query A (`es`): cuando salió el troll de la película
- query B (`zh`): 电影巨魔什么时候出来的
- diagnosis: RecallDrop; nDCG@10 end=38.6853, mix=0.0000, Δ=-38.6853; Recall@10 end=100.0000, mix=0.0000, Δ=-100.0000; tokens(a/b)=7/6, len_ratio=1.1667; overlap@10=9; source=evaluate_perquery

Endpoint top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 3428904 | 0 | 0.6586 | es | Troll (película) No confundir con Trolls (película). Troll es una película de fantasía de comedia oscura de culto de 1986 dirigida por John Carl Buechler y producida por Charles... |
| 2 | 3596383 | 0 | 0.6565 | es | Troll (película) De Wikipedia, la enciclopedia libre. Para ver la próxima película de DreamWorks, vea Trolls (película). Troll es una película de fantasía de comedia oscura de c... |
| 3 | 3596376 | 0 | 0.6481 | es | Troll (película) Para la próxima película de DreamWorks, vea Trolls (película). Troll es una película de fantasía de comedia oscura de culto de 1986 dirigida por John Carl Buech... |
| 4 | 7805649 | 0 | 0.6451 | es | Troll (película) Para la próxima película de DreamWorks, vea Trolls (película). Troll es una película de fantasía de comedia oscura de culto de 1986 dirigida por John Carl Buech... |
| 5 | 7253932 | 1 | 0.6448 | es | Troll se estrenó en 959 pantallas en enero de 1986 y, en su primera semana de lanzamiento, ocupó el noveno lugar en la lista de taquilla con una recaudación de $ 2.1 millones. E... |
| 6 | 4350821 | 0 | 0.6419 | es | No confundir con Trolls (película). Troll es una película de fantasía de comedia oscura de culto de 1986 dirigida por John Carl Buechler y producida por Charles Band of Empire P... |
| 7 | 2964260 | 0 | 0.6405 | zh | The Boss Baby (2017) Trolls 2 (2020) Trolls 是一部 2016 年美国 3D 电脑动画音乐喜剧电影，由迈克·米切尔执导，梦工厂动画制作，20 世纪福克斯发行。这部电影将由安娜·肯德里克和贾斯汀·汀布莱克主演。 |
| 8 | 3405847 | 0 | 0.6398 | es | Troll (1986) PG-13 \| 1h 22min \| Comedia, Fantasía, Terror \| 17 de enero de 1986 (EE. UU.) Un malvado rey trol en busca de un anillo místico que lo devuelva a su forma humana ... |
| 9 | 3916485 | 0 | 0.6371 | es | Troll es una película de culto de fantasía oscura de 1986 dirigida por John Carl Buechler y producida por Charles Band of Empire Pictures, protagonizada por Noah Hathaway, Micha... |
| 10 | 3916480 | 0 | 0.6261 | es | Troll fue lanzado en un DVD de doble función con Troll 2 por MGM el 26 de agosto de 2003. Scream Factory lanzará un Blu-ray de doble función de Troll y Troll 2 el 17 de noviembr... |

Mixed top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 3405847 | 0 | 0.6626 | es | Troll (1986) PG-13 \| 1h 22min \| Comedia, Fantasía, Terror \| 17 de enero de 1986 (EE. UU.) Un malvado rey trol en busca de un anillo místico que lo devuelva a su forma humana ... |
| 2 | 3596383 | 0 | 0.6581 | es | Troll (película) De Wikipedia, la enciclopedia libre. Para ver la próxima película de DreamWorks, vea Trolls (película). Troll es una película de fantasía de comedia oscura de c... |
| 3 | 3596376 | 0 | 0.6580 | es | Troll (película) Para la próxima película de DreamWorks, vea Trolls (película). Troll es una película de fantasía de comedia oscura de culto de 1986 dirigida por John Carl Buech... |
| 4 | 2964260 | 0 | 0.6537 | zh | The Boss Baby (2017) Trolls 2 (2020) Trolls 是一部 2016 年美国 3D 电脑动画音乐喜剧电影，由迈克·米切尔执导，梦工厂动画制作，20 世纪福克斯发行。这部电影将由安娜·肯德里克和贾斯汀·汀布莱克主演。 |
| 5 | 7805649 | 0 | 0.6518 | es | Troll (película) Para la próxima película de DreamWorks, vea Trolls (película). Troll es una película de fantasía de comedia oscura de culto de 1986 dirigida por John Carl Buech... |
| 6 | 3428904 | 0 | 0.6513 | es | Troll (película) No confundir con Trolls (película). Troll es una película de fantasía de comedia oscura de culto de 1986 dirigida por John Carl Buechler y producida por Charles... |
| 7 | 3916485 | 0 | 0.6426 | es | Troll es una película de culto de fantasía oscura de 1986 dirigida por John Carl Buechler y producida por Charles Band of Empire Pictures, protagonizada por Noah Hathaway, Micha... |
| 8 | 6116933 | 0 | 0.6424 | zh | Trolls the Musical (episode) Trolls the Musical 是 Harry and Friends 第一季的第 17 集和第二部电视电影。它于 1995 年 8 月 18 日播出，并于 1995 年 8 月 22 日在 VHS、2000 年 5 月 23 日和 2008 年 11 月 4 日发行了 DVD。八年后的续... |
| 9 | 3916480 | 0 | 0.6416 | es | Troll fue lanzado en un DVD de doble función con Troll 2 por MGM el 26 de agosto de 2003. Scream Factory lanzará un Blu-ray de doble función de Troll y Troll 2 el 17 de noviembr... |
| 10 | 4350821 | 0 | 0.6403 | es | No confundir con Trolls (película). Troll es una película de fantasía de comedia oscura de culto de 1986 dirigida por John Carl Buechler y producida por Charles Band of Empire P... |

### qid `290830`

- query A (`es`): ¿cuántos oscars tiene peter jackson wolllnlll? l
- query B (`zh`): 彼得·杰克逊·沃尔获得了多少奥斯卡奖？l
- diagnosis: RecallDrop; nDCG@10 end=38.6853, mix=0.0000, Δ=-38.6853; Recall@10 end=100.0000, mix=0.0000, Δ=-100.0000; tokens(a/b)=7/9, len_ratio=0.7778; overlap@10=7; source=evaluate_perquery

Endpoint top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 2482281 | 0 | 0.6219 | es | Peter Jackson ha sido nominado 8 veces como productor, director y guionista. Ganó las tres categorías en 2003 por El señor de los anillos: El. |
| 2 | 1177372 | 0 | 0.6102 | zh | 彼得杰克逊凭借《指环王：王者归来》获得奥斯卡奖 |
| 3 | 1069019 | 0 | 0.6086 | es | ¿Cuántos premios Oscar ha ganado Peter Jackson? Peter Jackson ha sido nominado 8 veces como productor, director y guionista. Ganó las tres categorías en 2003 por El Señor de los... |
| 4 | 2482276 | 0 | 0.5991 | es | Peter Jackson ha sido nominado 8 veces como productor, director y guionista. Ganó las tres categorías en 2003 por El Señor de los Anillos: El Retorno del Rey. Tres . Mejor direc... |
| 5 | 691855 | 1 | 0.5972 | es | Peter Jackson, nativo de Nueva Zelanda, es mejor conocido como director por su adaptación de J.R.R. La trilogía El señor de los anillos de Tolkien, que ganó 11 premios Oscar. Si... |
| 6 | 2482277 | 0 | 0.5891 | es | Peter Jackson ha sido nominado 8 veces como productor, director y guionista. Ganó las tres categorías en 2003 por El Señor de los Anillos: El Retorno del Rey. Peter Jackson ha s... |
| 7 | 6660330 | 0 | 0.5807 | zh | 威廉姆斯曾获得五项奥斯卡金像奖、四项金球奖、七项英国电影学院奖和 22 项格莱美奖。威廉姆斯获得 49 项奥斯卡奖提名，是仅次于沃尔特迪斯尼的第二大提名人。 |
| 8 | 7377664 | 0 | 0.5795 | zh | 约翰·威廉姆斯获得奥斯卡奖提名的次数高达 49 次，其中包括他 2013 年为《偷书贼》获得的原创评分。在奥斯卡历史上，只有沃尔特·迪斯尼 (Walt Disney) 获得了 59 次提名，获得了更多提名提名。威廉姆斯凭借他的电影配乐获得了五项奥斯卡奖。 |
| 9 | 1171089 | 0 | 0.5757 | es | La trilogía cinematográfica El señor de los anillos, dirigida por Peter Jackson, ha ganado diecisiete premios Oscar y fue nominada a otros trece premios de la Academia. |
| 10 | 6584743 | 0 | 0.5755 | zh | 约翰威廉姆斯获得 50 项奥斯卡奖提名，获得 5 项； 6 次艾美奖，获得 3 次； 25 次金球奖，获得 4 次； 67 次格莱美奖，获得 23 次；并获得了 7 项英国电影学院奖。 |

Mixed top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 1177372 | 0 | 0.6495 | zh | 彼得杰克逊凭借《指环王：王者归来》获得奥斯卡奖 |
| 2 | 2482281 | 0 | 0.6356 | es | Peter Jackson ha sido nominado 8 veces como productor, director y guionista. Ganó las tres categorías en 2003 por El señor de los anillos: El. |
| 3 | 1069019 | 0 | 0.6249 | es | ¿Cuántos premios Oscar ha ganado Peter Jackson? Peter Jackson ha sido nominado 8 veces como productor, director y guionista. Ganó las tres categorías en 2003 por El Señor de los... |
| 4 | 7377664 | 0 | 0.6167 | zh | 约翰·威廉姆斯获得奥斯卡奖提名的次数高达 49 次，其中包括他 2013 年为《偷书贼》获得的原创评分。在奥斯卡历史上，只有沃尔特·迪斯尼 (Walt Disney) 获得了 59 次提名，获得了更多提名提名。威廉姆斯凭借他的电影配乐获得了五项奥斯卡奖。 |
| 5 | 6660330 | 0 | 0.6140 | zh | 威廉姆斯曾获得五项奥斯卡金像奖、四项金球奖、七项英国电影学院奖和 22 项格莱美奖。威廉姆斯获得 49 项奥斯卡奖提名，是仅次于沃尔特迪斯尼的第二大提名人。 |
| 6 | 2482276 | 0 | 0.6129 | es | Peter Jackson ha sido nominado 8 veces como productor, director y guionista. Ganó las tres categorías en 2003 por El Señor de los Anillos: El Retorno del Rey. Tres . Mejor direc... |
| 7 | 1732202 | 0 | 0.6048 | zh | 沃尔特在他的职业生涯中个人赢得了 32 项奥斯卡奖。直到今天，沃尔特仍然保持着个人获得奥斯卡奖最多的记录，而且只有一个地方可以让您近距离观察 20 多个著名的小雕像。 ‚â€ 华特迪士尼家族博物馆。 |
| 8 | 2482277 | 0 | 0.6045 | es | Peter Jackson ha sido nominado 8 veces como productor, director y guionista. Ganó las tres categorías en 2003 por El Señor de los Anillos: El Retorno del Rey. Peter Jackson ha s... |
| 9 | 1596740 | 0 | 0.6038 | zh | 沃尔特迪斯尼（1901 年）（1901 年）获得或获得了 26 项奥斯卡金像奖，并保持着历史上奥斯卡奖最多的记录。他一共获得了 59 项提名，共获得了 22 项竞争性奥斯卡奖，并保持着历史上个人获奖最多和提名最多的记录。 迪斯尼赢得了他的第一个竞争性奥斯卡奖并获得了他的第一个荣誉学院第五届奥斯卡金像奖（1932年）。他因创作米老鼠而获得奥斯卡荣誉奖，并... |
| 10 | 7556946 | 0 | 0.6038 | zh | 威廉姆斯曾获得五项奥斯卡金像奖、四项金球奖、七项英国电影学院奖和 22 项格莱美奖。威廉姆斯获得 49 项奥斯卡奖提名，是继沃尔特·迪斯尼 (Walt Disney.illiams) 获得五项奥斯卡金像奖、四项金球奖、七项英国电影学院奖和 22 项格莱美奖之后，获得第二多提名的个人。威廉姆斯获得 49 项奥斯卡奖提名，是仅次于沃尔特迪斯尼的第二大提名人。 |

### qid `1000678`

- query A (`es`): ¿Dónde exploró el capitán James Cook?
- query B (`zh`): 詹姆斯库克船长去哪里探索了？
- diagnosis: RankDrop; nDCG@10 end=100.0000, mix=63.0930, Δ=-36.9070; Recall@10 end=100.0000, mix=100.0000, Δ=0.0000; tokens(a/b)=6/6, len_ratio=1.0000; overlap@10=10; source=evaluate_perquery

Endpoint top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 7262611 | 1 | 0.6812 | es | Cook, James (1728ÃƒÂ ¢ Ã‚â‚¬Ã‚â € œ1779) Un navegante, topógrafo y explorador, que fue capitán de tres expediciones al Pacífico entre 1768 y 1779. Estudió las costas de Nueva Ze... |
| 2 | 7262614 | 0 | 0.6774 | es | Cook, James (1728-1779) explorador inglés. James Cook fue una de las figuras más destacadas de la Era de la Exploración. Durante su carrera, Cook dio la vuelta al mundo dos vece... |
| 3 | 576850 | 0 | 0.6771 | es | James Cook (1728-1779) Famoso por: Descubrimiento de las islas hawaianas. Un capitán inglés que dio la vuelta al Atlántico hasta el Océano Pacífico. Se puso en contacto con los ... |
| 4 | 2619434 | 0 | 0.6758 | es | El primer barco de James Cook, en el que circunnavegó Nueva Zelanda y atravesó la costa este de Australia, se llamó HMS Endeavour. La mayor exploración del capitán James Cook fu... |
| 5 | 8775163 | 0 | 0.6661 | es | Ruta del primer viaje de James Cook. El primer viaje de James Cook fue una expedición combinada de la Royal Navy y la Royal Society al Océano Pacífico sur a bordo del HMS Endeav... |
| 6 | 8775162 | 0 | 0.6657 | es | James Cook 1728-1779 James Cook nació en el pueblo de Marton, en Yorkshire, el 27 de octubre de 1728. Su primera experiencia en el mar se produjo a la edad de 18 años, cuando se... |
| 7 | 739573 | 0 | 0.6639 | es | Nacido el 27 de octubre de 1728 en Marton-in-Cleveland, Yorkshire, Inglaterra, James Cook fue un capitán naval, navegante y explorador que, en 1770, descubrió y cartografió Nuev... |
| 8 | 4075458 | 0 | 0.6625 | es | Ocupación: Explorador; Nacimiento: 27 de octubre de 1728 en Marton, Inglaterra; Muerto: Asesinado por nativos en las islas hawaianas el 14 de febrero de 1779; Mejor conocido por... |
| 9 | 8775160 | 0 | 0.6609 | es | El capitán James Cook enfrentó problemas en sus viajes, particularmente en su primer viaje importante a Nueva Zelanda y la costa este de Australia. Mientras estaba en Tahití, la... |
| 10 | 8775161 | 0 | 0.6590 | es | El capitán James Cook enfrentó algunos problemas en sus viajes, particularmente su primer viaje importante a Nueva Zelanda y la costa este de Australia. Mientras estaba en Tahit... |

Mixed top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 2619434 | 0 | 0.6959 | es | El primer barco de James Cook, en el que circunnavegó Nueva Zelanda y atravesó la costa este de Australia, se llamó HMS Endeavour. La mayor exploración del capitán James Cook fu... |
| 2 | 7262611 | 1 | 0.6950 | es | Cook, James (1728ÃƒÂ ¢ Ã‚â‚¬Ã‚â € œ1779) Un navegante, topógrafo y explorador, que fue capitán de tres expediciones al Pacífico entre 1768 y 1779. Estudió las costas de Nueva Ze... |
| 3 | 576850 | 0 | 0.6875 | es | James Cook (1728-1779) Famoso por: Descubrimiento de las islas hawaianas. Un capitán inglés que dio la vuelta al Atlántico hasta el Océano Pacífico. Se puso en contacto con los ... |
| 4 | 7262614 | 0 | 0.6856 | es | Cook, James (1728-1779) explorador inglés. James Cook fue una de las figuras más destacadas de la Era de la Exploración. Durante su carrera, Cook dio la vuelta al mundo dos vece... |
| 5 | 8775163 | 0 | 0.6854 | es | Ruta del primer viaje de James Cook. El primer viaje de James Cook fue una expedición combinada de la Royal Navy y la Royal Society al Océano Pacífico sur a bordo del HMS Endeav... |
| 6 | 8775160 | 0 | 0.6809 | es | El capitán James Cook enfrentó problemas en sus viajes, particularmente en su primer viaje importante a Nueva Zelanda y la costa este de Australia. Mientras estaba en Tahití, la... |
| 7 | 8775161 | 0 | 0.6803 | es | El capitán James Cook enfrentó algunos problemas en sus viajes, particularmente su primer viaje importante a Nueva Zelanda y la costa este de Australia. Mientras estaba en Tahit... |
| 8 | 8775162 | 0 | 0.6776 | es | James Cook 1728-1779 James Cook nació en el pueblo de Marton, en Yorkshire, el 27 de octubre de 1728. Su primera experiencia en el mar se produjo a la edad de 18 años, cuando se... |
| 9 | 739573 | 0 | 0.6764 | es | Nacido el 27 de octubre de 1728 en Marton-in-Cleveland, Yorkshire, Inglaterra, James Cook fue un capitán naval, navegante y explorador que, en 1770, descubrió y cartografió Nuev... |
| 10 | 4075458 | 0 | 0.6718 | es | Ocupación: Explorador; Nacimiento: 27 de octubre de 1728 en Marton, Inglaterra; Muerto: Asesinado por nativos en las islas hawaianas el 14 de febrero de 1779; Mejor conocido por... |

### qid `1002238`

- query A (`es`): ¿Cuándo se pusieron los guepardos en la lista de especies en peligro de extinción?
- query B (`zh`): 猎豹什么时候被列入濒危名单
- diagnosis: TranslationDivergence; nDCG@10 end=100.0000, mix=63.0930, Δ=-36.9070; Recall@10 end=100.0000, mix=100.0000, Δ=0.0000; tokens(a/b)=14/7, len_ratio=2.0000; overlap@10=9; source=evaluate_perquery

Endpoint top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 7260660 | 1 | 0.6615 | es | Los guepardos deberían estar en la lista de especies en peligro de extinción El primero surgió después de analizar más de 20.000 observaciones de turistas y dos millones de dato... |
| 2 | 7260665 | 0 | 0.6571 | es | Lamentablemente, los científicos están tratando de convencer a los legisladores de que incluyan a los guepardos en la lista de especies en peligro de extinción, ya que su poblac... |
| 3 | 5149352 | 0 | 0.6459 | es | Los guepardos figuran actualmente como "vulnerables" en la lista roja de la Unión Internacional para la Conservación de la Naturaleza, que supervisa el número de poblaciones de ... |
| 4 | 7260659 | 0 | 0.6455 | es | Los científicos están haciendo todo lo posible para incluir a los guepardos en la lista de especies en peligro de extinción. Un grupo de investigadores se ha unido para pedirle ... |
| 5 | 5149359 | 0 | 0.6448 | es | El USFWS enumera al guepardo como en peligro de extinción y la Lista Roja de la UICN los identifica como vulnerables, que enfrentan un alto riesgo de extinción en la naturaleza.... |
| 6 | 4459107 | 0 | 0.6319 | es | En un censo de 1900, la población de guepardos era de alrededor de 100.000. Hoy en día, solo quedan 9.000 en África. Con menos presas y hábitat… y perseguido por cazadores… el g... |
| 7 | 6577772 | 0 | 0.6289 | es | Hoy, el guepardo ha sido catalogado por la UICN como una especie vulnerable a la extinción en su entorno natural en un futuro próximo. La pérdida de hábitat junto con el aumento... |
| 8 | 7452003 | 0 | 0.6228 | es | En un censo de 1900, la población de guepardos era de alrededor de 100.000. Hoy en día, solo quedan 9.000 en África. Con menos presas y hábitat, y perseguido por cazadores, el g... |
| 9 | 3284412 | 0 | 0.6176 | es | Estos son algunos de los datos sobre guepardos más interesantes y sorprendentes para los niños. El guepardo (Acinonyx jubatus) es un felino endémico de África con algunas de las... |
| 10 | 7452001 | 0 | 0.6174 | es | HECHOS DEL GUEPARDO. El guepardo es posiblemente uno de los más bellos y atléticos de todos los grandes felinos, pero también es uno de los más amenazados. La población estimada... |

Mixed top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 5149352 | 0 | 0.6620 | es | Los guepardos figuran actualmente como "vulnerables" en la lista roja de la Unión Internacional para la Conservación de la Naturaleza, que supervisa el número de poblaciones de ... |
| 2 | 7260660 | 1 | 0.6556 | es | Los guepardos deberían estar en la lista de especies en peligro de extinción El primero surgió después de analizar más de 20.000 observaciones de turistas y dos millones de dato... |
| 3 | 7260665 | 0 | 0.6514 | es | Lamentablemente, los científicos están tratando de convencer a los legisladores de que incluyan a los guepardos en la lista de especies en peligro de extinción, ya que su poblac... |
| 4 | 5149359 | 0 | 0.6492 | es | El USFWS enumera al guepardo como en peligro de extinción y la Lista Roja de la UICN los identifica como vulnerables, que enfrentan un alto riesgo de extinción en la naturaleza.... |
| 5 | 7260659 | 0 | 0.6405 | es | Los científicos están haciendo todo lo posible para incluir a los guepardos en la lista de especies en peligro de extinción. Un grupo de investigadores se ha unido para pedirle ... |
| 6 | 4459107 | 0 | 0.6311 | es | En un censo de 1900, la población de guepardos era de alrededor de 100.000. Hoy en día, solo quedan 9.000 en África. Con menos presas y hábitat… y perseguido por cazadores… el g... |
| 7 | 6577772 | 0 | 0.6279 | es | Hoy, el guepardo ha sido catalogado por la UICN como una especie vulnerable a la extinción en su entorno natural en un futuro próximo. La pérdida de hábitat junto con el aumento... |
| 8 | 5149355 | 0 | 0.6206 | es | Ese número es la población más grande de guepardos en libertad en el mundo, y menos de la mitad de la estimación publicada en noviembre de 2016. Tal como está, la especie figura... |
| 9 | 7452001 | 0 | 0.6206 | es | HECHOS DEL GUEPARDO. El guepardo es posiblemente uno de los más bellos y atléticos de todos los grandes felinos, pero también es uno de los más amenazados. La población estimada... |
| 10 | 7452003 | 0 | 0.6195 | es | En un censo de 1900, la población de guepardos era de alrededor de 100.000. Hoy en día, solo quedan 9.000 en África. Con menos presas y hábitat, y perseguido por cazadores, el g... |

### qid `1005653`

- query A (`es`): ¿Cuándo dirigió William Whyte su estudio?
- query B (`zh`): 威廉·怀特什么时候进行他的研究
- diagnosis: RankDrop; nDCG@10 end=100.0000, mix=63.0930, Δ=-36.9070; Recall@10 end=100.0000, mix=100.0000, Δ=0.0000; tokens(a/b)=6/8, len_ratio=0.7500; overlap@10=5; source=evaluate_perquery

Endpoint top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 7254368 | 1 | 0.5460 | es | Whyte nació en West Chester, Pensilvania, y se graduó de la Universidad de Princeton en 1939. En 1946, ocupó un puesto en la revista Fortune, donde ganó reconocimiento como teór... |
| 2 | 7254370 | 0 | 0.5185 | es | La Comisión de Planificación de la alcaldesa Lindsay le pidió a William Whyte, un urbanista y sociólogo que había trabajado anteriormente con la Comisión en el Plan de 1969 para... |
| 3 | 7254375 | 0 | 0.5044 | es | A su muerte, William F. Whyte fue sobrevivido por su esposa, Kathleen (King) Whyte, dos hijos y dos hijas. Educación. El profesor Whyte recibió su licenciatura en economía de Sw... |
| 4 | 7254366 | 0 | 0.5037 | es | Las ideas de Whyte son tan relevantes hoy como lo eran hace más de 30 años, y quizás incluso más. Biografía. Whyte nació en West Chester, Pensilvania, en 1917. Después de gradua... |
| 5 | 7884343 | 0 | 0.4972 | es | David Whyte es miembro asociado de SaÃƒÆ'Ã‚Â¯d Business School en la Universidad de Oxford. Sus libros incluyen The Heart Aroused: Poetry and the Preservation of the Soul in Cor... |
| 6 | 7254367 | 0 | 0.4970 | es | En febrero de 1937, mientras estaba en Harvard, Whyte comenzó a investigar sobre el libro que haría su nombre, alquilando una habitación en el tercer piso en el North End, enton... |
| 7 | 7884345 | 0 | 0.4951 | es | ÃƒÂ ¢ Ã‚â‚¬Ã‚Å “FinisterreÃƒÂ ¢ Ã‚â‚¬Ã‚. David Whyte. es miembro asociado de SaÃƒÆ’Ã‚Â¯d Business School en la Universidad de Oxford. Sus libros incluyen The Heart Aroused: Poet... |
| 8 | 3315936 | 0 | 0.4906 | zh | 威廉·马克西米利安·冯特。威廉·马克西米利安·冯特（Wilhelm Maximilian Wundt，1832 年）被后人称为“实验心理学之父”和“实验心理学之父”。第一个心理学实验室（Boring 1950: 317, 322, 344 â€œ5），从那里他对心理学作为一门学科的发展产生了巨大的影响，尤其是在美国。在公共场合保留和害羞（参见 |
| 9 | 5041817 | 0 | 0.4865 | zh | 威廉詹姆斯在哈佛任教，并于 1878 年撰写了《心理学原理》。这是第一本《心理学导论》。詹姆斯于 1875 年（比冯特的 1879 年早四年）建立了一个心理实验室，但主要是为了教学演示。 |
| 10 | 5888367 | 0 | 0.4795 | es | Hooke estudió en Wadham College durante el Protectorado, donde se convirtió en uno de un grupo muy unido de ardientes realistas dirigidos por John Wilkins. Aquí fue empleado com... |

Mixed top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 7254370 | 0 | 0.5715 | zh | 林赛市长的规划委员会请城市学家和社会学家威廉·怀特 (William Whyte) 进行这项研究。这篇文章认为，怀特之前在林赛的规划委员会中的经历形成了他的方法论方法。 |
| 2 | 7254368 | 1 | 0.5644 | es | Whyte nació en West Chester, Pensilvania, y se graduó de la Universidad de Princeton en 1939. En 1946, ocupó un puesto en la revista Fortune, donde ganó reconocimiento como teór... |
| 3 | 3315936 | 0 | 0.5513 | zh | 威廉·马克西米利安·冯特。威廉·马克西米利安·冯特（Wilhelm Maximilian Wundt，1832 年）被后人称为“实验心理学之父”和“实验心理学之父”。第一个心理学实验室（Boring 1950: 317, 322, 344 â€œ5），从那里他对心理学作为一门学科的发展产生了巨大的影响，尤其是在美国。在公共场合保留和害羞（参见 |
| 4 | 5041817 | 0 | 0.5500 | zh | 威廉詹姆斯在哈佛任教，并于 1878 年撰写了《心理学原理》。这是第一本《心理学导论》。詹姆斯于 1875 年（比冯特的 1879 年早四年）建立了一个心理实验室，但主要是为了教学演示。 |
| 5 | 6710993 | 0 | 0.5455 | zh | 威廉 H. 怀特是经典社会学评论组织人的作者，他在 1952 年发表在《财富》杂志上的一篇文章中创造了“群体思维”一词，提及二战后塑造了许多公司的从众文化。经典社会学评论组织人，在 1952 年发表在《财富》杂志上的一篇文章中创造了团体思维一词，该文章提到了二战后塑造了许多公司的从众文化。 |
| 6 | 2762115 | 0 | 0.5414 | zh | 威廉·冯特（Wilhelm Wundt，1832 年 8 月 16 日出生于内卡劳，靠近曼海姆，巴登 [德国] â€€ 死于 1920 年 8 月 31 日，德国格罗斯博滕），德国生理学家和心理学家，公认的实验心理学的创始人。 1 冯特于 1856 年在海德堡大学获得医学学位。在与 Johannes MÃƒÆ'Ã‚Â¼ller 短暂学习后，他被任命为海... |
| 7 | 7254367 | 0 | 0.5412 | es | En febrero de 1937, mientras estaba en Harvard, Whyte comenzó a investigar sobre el libro que haría su nombre, alquilando una habitación en el tercer piso en el North End, enton... |
| 8 | 3114247 | 0 | 0.5378 | zh | 1851年威廉·冯特于1856年在海德堡大学学习。在他的整个职业生涯中，他于1874年在苏黎世大学担任归纳哲学教授。从1875年到1917年，他在莱比锡大学担任归纳哲学教授。他的一些学生包括 J.M. Cattell、Titchener 和 Spearman。 |
| 9 | 4794448 | 0 | 0.5363 | zh | Wilhelm Maximilian Wundt (1832-1920) 被后人称为“实验心理学之父”和第一个心理学实验室的创始人（Boring，1950： 317、322、344-5)。从那里，冯特对心理学作为一门学科的发展产生了巨大的影响，尤其是在美国。 伊尔海姆·马克西米利安·冯特于 1832 年 8 月 16 日出生在曼海姆郊外的内卡劳镇，是路... |
| 10 | 5972144 | 0 | 0.5362 | zh | 威廉·冯特通常被认为是实验心理学的创始人。 1862 年，他教授了第一门生理心理学课程，并出版了《生理心理学原理》，这是第一本将心理学确立为自己领域的著作，从而为心理学做出了贡献。 |

### qid `1032019`

- query A (`es`): ¿Cuál es la longitud estándar del cañón para un ar?
- query B (`zh`): ar 的标准枪管长度是多少？
- diagnosis: RankDrop; nDCG@10 end=100.0000, mix=63.0930, Δ=-36.9070; Recall@10 end=100.0000, mix=100.0000, Δ=0.0000; tokens(a/b)=10/7, len_ratio=1.4286; overlap@10=7; source=evaluate_perquery

Endpoint top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 7214862 | 1 | 0.6808 | es | Longitud del cañón AR-15. El AR-15 es quizás la carabina más popular en Estados Unidos. Comúnmente se compara con el AK-47 a pesar de la gran cantidad de diferencias entre las d... |
| 2 | 7214857 | 0 | 0.6761 | es | Longitud del cañón AR-15: ¿Qué longitud debe elegir? ¿Es un barril AR más largo más preciso? Para mucha gente, es simple: un cañón más largo dispara con mayor precisión que uno ... |
| 3 | 2934424 | 0 | 0.6575 | es | La caja del cartucho no se basa en el 5.56 mm, sino en un .30 Remington, por lo que para convertir un AR estándar, estamos hablando de reemplazar el cerrojo, el cañón y el carga... |
| 4 | 1657558 | 0 | 0.6499 | es | Respuestas. Mejor respuesta: la longitud estándar de un cañón m4 es de aproximadamente 363 mm. Supongo que tiene la variante M4 CQB que es de alrededor de 300 mm. Así que realme... |
| 5 | 7214860 | 0 | 0.6417 | zh | 绝大多数 AR 枪管介于 14.5 军用 M4 型和 20. M16 尺寸之间，其中 16 型目前最受平民欢迎。您可以购买 14.5 桶，但对我们大多数人来说，它需要一个至少 1.5 英寸长且永久固定的枪口装置，才能满足法定的 16 最低要求。 |
| 6 | 4485132 | 0 | 0.6319 | es | Una arena de tamaño estándar. Una arena de tamaño estándar tiene 130 pies de ancho por 200 pies de largo, por lo que las distancias de los cañones son las siguientes: 60 pies de... |
| 7 | 7214859 | 0 | 0.6274 | es | Para resumir la longitud del cañón: para la mayoría de las personas, es un cañón de 14,5 (+1,5) o 16 para un manejo más fácil en espacios reducidos, o un cañón de 18 o 20 para u... |
| 8 | 7214861 | 0 | 0.6241 | es | El sistema de longitud media (por cierto, no es un estándar oficial, pero de todos modos es un estándar de la industria) se encuentra comúnmente en 18 cañones, pero se pueden ob... |
| 9 | 6652351 | 0 | 0.6207 | es | la longitud del cañón es desde la base del caparazón hasta la boca del cañón. el mono va desde la culata hasta la boca del arma. incluso si el arma es una milésima de pulgada de... |
| 10 | 2434096 | 0 | 0.6180 | es | La longitud de un palo que se adapta a un jugador no necesariamente se adapta a otro, razón por la cual hay montadores profesionales de palos. Con respecto a los palos estándar,... |

Mixed top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 7214860 | 0 | 0.7118 | zh | 绝大多数 AR 枪管介于 14.5 军用 M4 型和 20. M16 尺寸之间，其中 16 型目前最受平民欢迎。您可以购买 14.5 桶，但对我们大多数人来说，它需要一个至少 1.5 英寸长且永久固定的枪口装置，才能满足法定的 16 最低要求。 |
| 2 | 7214862 | 1 | 0.6961 | es | Longitud del cañón AR-15. El AR-15 es quizás la carabina más popular en Estados Unidos. Comúnmente se compara con el AK-47 a pesar de la gran cantidad de diferencias entre las d... |
| 3 | 2934424 | 0 | 0.6866 | es | La caja del cartucho no se basa en el 5.56 mm, sino en un .30 Remington, por lo que para convertir un AR estándar, estamos hablando de reemplazar el cerrojo, el cañón y el carga... |
| 4 | 7214857 | 0 | 0.6816 | es | Longitud del cañón AR-15: ¿Qué longitud debe elegir? ¿Es un barril AR más largo más preciso? Para mucha gente, es simple: un cañón más largo dispara con mayor precisión que uno ... |
| 5 | 5874334 | 0 | 0.6433 | es | Si bien se supone que el rifle tipo AR en calibres tradicionales es un cartucho de 400 yardas o menos, algunas personas emprendedoras han convertido una variante en un ejecutant... |
| 6 | 7214861 | 0 | 0.6383 | es | El sistema de longitud media (por cierto, no es un estándar oficial, pero de todos modos es un estándar de la industria) se encuentra comúnmente en 18 cañones, pero se pueden ob... |
| 7 | 4669835 | 0 | 0.6328 | es | AMO CONVENCIONAL. ESTÁNDAR DE LONGITUD DEL ARCO. El estándar de longitud de arco AMO está diseñado para ser tres pulgadas más largo que la cuerda de arco AMO. Maestro que apunta... |
| 8 | 6062645 | 0 | 0.6307 | es | La longitud mínima del cañón para las escopetas en la mayor parte de los EE. UU. Es de 18 pulgadas (460 mm), y esta longitud del cañón (a veces 18.5ÃƒÂ ¢ Ã‚â‚¬Ã‚â € œ20 in (470Ã... |
| 9 | 2434096 | 0 | 0.6304 | es | La longitud de un palo que se adapta a un jugador no necesariamente se adapta a otro, razón por la cual hay montadores profesionales de palos. Con respecto a los palos estándar,... |
| 10 | 1657558 | 0 | 0.6297 | es | Respuestas. Mejor respuesta: la longitud estándar de un cañón m4 es de aproximadamente 363 mm. Supongo que tiene la variante M4 CQB que es de alrededor de 300 mm. Así que realme... |

### qid `1035874`

- query A (`es`): ¿Cuál es el sólido más importante disuelto en el agua del océano?
- query B (`zh`): 溶解在海水中的最重要的固体是什么？
- diagnosis: RankDrop; nDCG@10 end=100.0000, mix=63.0930, Δ=-36.9070; Recall@10 end=100.0000, mix=100.0000, Δ=0.0000; tokens(a/b)=12/11, len_ratio=1.0909; overlap@10=7; source=evaluate_perquery

Endpoint top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 7208363 | 1 | 0.6626 | es | Los seis sólidos disueltos más comunes en el agua del océano son _____, _____, _____, _____, _____. cloro / sodio / magnesio / azufre / calcio / potasio La mayoría de los elemen... |
| 2 | 7208366 | 0 | 0.6442 | es | Answers.com Ãƒâ € šÃ‚Â® WikiAnswers Ãƒâ € šÃ‚Â® Categorías Ciencia Geografía Cuerpos de agua Océanos y mares ¿Cuál es el sólido más disuelto en el océano? |
| 3 | 5776202 | 0 | 0.6266 | es | ¿En qué se diferencian las moléculas de un sólido de las de un líquido o gas? |
| 4 | 5070210 | 0 | 0.6206 | es | La sal disuelta más abundante en el agua del océano es el cloruro de sodio. Es un compuesto cristalino incoloro con la fórmula química de NaCl. |
| 5 | 5070206 | 0 | 0.6206 | es | La sal disuelta más abundante en el agua del océano es el cloruro de sodio. Es un compuesto cristalino incoloro con la fórmula química de NaCl. |
| 6 | 1577014 | 0 | 0.6205 | es | La sal es, por supuesto, la sustancia más común que se encuentra en el agua de mar, pero está lejos de ser la única. De hecho, el agua de mar es bastante rica en sólidos mineral... |
| 7 | 1151082 | 0 | 0.6202 | es | Los iones más abundantes disueltos en el agua de mar son el cloruro, el sodio y el sulfato. La cantidad de sólidos inorgánicos disueltos en agua es su salinidad. |
| 8 | 1788948 | 0 | 0.6163 | es | Los sólidos disueltos son exactamente lo que parece. Los sólidos disueltos son diferentes minerales disueltos en agua. No son un componente específico, sino una combinación de d... |
| 9 | 3802452 | 0 | 0.6042 | es | Por qué es importante el oxígeno disuelto. El oxígeno disuelto (OD) es oxígeno disuelto en agua. El oxígeno se disuelve por difusión del aire circundante; aireación del agua que... |
| 10 | 6788692 | 0 | 0.6034 | es | Sólidos disueltos totales. El total de sólidos disueltos es una medida de todo lo que alguna vez se ha disuelto en el agua de su piscina. Esto incluye minerales que se separan d... |

Mixed top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 7208366 | 0 | 0.6616 | zh | Answers.com Ãƒâ€šÃ‚Â® WikiAnswers Ãƒâ€šÃ‚Â® Categories 科学地理 水体 海洋 海洋中溶解度最高的固体是什么？ |
| 2 | 7208363 | 1 | 0.6584 | es | Los seis sólidos disueltos más comunes en el agua del océano son _____, _____, _____, _____, _____. cloro / sodio / magnesio / azufre / calcio / potasio La mayoría de los elemen... |
| 3 | 1577014 | 0 | 0.6498 | zh | 当然，盐是海水中最常见的物质，但它远非唯一。事实上，海水实际上富含矿物质固体。溶解在海水中的固体含量从河流入海的约 1% 到水循环部分受限的约 4% 不等，但平均约 3.5%（100 磅海水，如果被煮沸，将留下接近 3.5 磅的固体）。其中大部分是普通盐——氯化钠。 |
| 4 | 5070210 | 0 | 0.6473 | zh | 海水中最丰富的溶解盐是氯化钠。它是一种无色结晶化合物，化学式为 NaCl。 |
| 5 | 5070206 | 0 | 0.6473 | zh | 海水中最丰富的溶解盐是氯化钠。它是一种无色结晶化合物，化学式为 NaCl。 |
| 6 | 1151079 | 0 | 0.6387 | zh | 显示更多结果。海水中最丰富的溶解离子是钠……钠离子和氯离子随后成为海盐中最丰富的成分。阅读更多。阳性：52%。海水化学。 ... 海水中的溶解盐。 1. 单位 = ppt 或 o/oo ... |
| 7 | 1151082 | 0 | 0.6385 | zh | 溶解在海水中的最丰富的离子是氯离子、钠离子和硫酸根离子。水中溶解的无机固体的数量就是它的盐度。 |
| 8 | 4800871 | 0 | 0.6338 | zh | 二氧化碳是溶解在海洋中的最重要的气体之一。其中一些以溶解气体的形式存在，但大多数与水反应形成碳酸或与水中已有的碳酸盐反应形成碳酸氢盐。这会从水中去除溶解的二氧化碳。互动：碳循环。二氧化碳是溶解在海洋中的最重要的气体之一。其中一些以溶解气体的形式存在，但大多数与水反应形成碳酸或与水中已有的碳酸盐反应形成碳酸氢盐。这可以从水中去除溶解的二氧化碳。 |
| 9 | 5776202 | 0 | 0.6258 | es | ¿En qué se diferencian las moléculas de un sólido de las de un líquido o gas? |
| 10 | 2663644 | 0 | 0.6246 | zh | 无机碳、溴化物、硼、锶和氟化物构成了海水的其他主要溶解物质，来自 http://www.britannica.com/EBchecked/topic/531121/seawater。所以无机碳是溶解在海水中最常见的非盐成分。 |

### qid `1036627`

- query A (`es`): cual es el significado de cruzamiento
- query B (`zh`): 杂交育种的意义是什么
- diagnosis: TranslationDivergence; nDCG@10 end=100.0000, mix=63.0930, Δ=-36.9070; Recall@10 end=100.0000, mix=100.0000, Δ=0.0000; tokens(a/b)=6/6, len_ratio=1.0000; overlap@10=6; source=evaluate_perquery

Endpoint top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 7207097 | 1 | 0.7312 | es | El cruzamiento significa la producción de un organismo mediante el apareamiento de dos especies, razas o variedades diferentes. También puede denominarse hibridación. El cruzami... |
| 2 | 7207104 | 0 | 0.7287 | es | El cruzamiento es el proceso de reproducción con la intención de crear descendencia que comparta los rasgos de ambos linajes parentales o para producir un animal con vigor híbri... |
| 3 | 2958424 | 0 | 0.6506 | es | cruzar, 1 Biología. 2 para cambiar la lealtad, como de un partido político a otro. 3 para cambiar con éxito de un campo de actividad, género, etc., a otro: pasar del jazz al roc... |
| 4 | 7024589 | 0 | 0.6474 | es | Cruzando. Definición. sustantivo. Un proceso que ocurre durante la meiosis en el que dos cromosomas se emparejan e intercambian segmentos de su material genético. Suplemento: es... |
| 5 | 2540596 | 0 | 0.6316 | es | Mejor respuesta: cruce es otro nombre para la recombinación o el intercambio físico de partes iguales de cromátidas adyacentes no hermanas. Cuando se produce el entrecruzamiento... |
| 6 | 2540593 | 0 | 0.6286 | es | cruzando. cruce, proceso en genética por el cual los dos cromosomas de un par homólogo intercambian segmentos iguales entre sí. El cruce ocurre en la primera división de la meio... |
| 7 | 2958426 | 0 | 0.6257 | es | cruzar, 1 Biología. (de un segmento de cromosoma) para someterse a un cruce. 2 para cambiar la lealtad, como de un partido político a otro. 3 para cambiar con éxito de un campo ... |
| 8 | 3752831 | 0 | 0.6255 | es | Definición de cruce para estudiantes del idioma inglés. : 1 un lugar donde se unen dos cosas. : 2 un lugar donde confluyen carreteras o vías férreas. : 3 una carretera o rampa q... |
| 9 | 4404143 | 0 | 0.6228 | es | DEFINICIÓN de 'Crossover' Un cruce es el punto en un gráfico de acciones cuando un valor y un indicador se cruzan. Los analistas técnicos utilizan cruces para ayudar a pronostic... |
| 10 | 4404141 | 0 | 0.6221 | es | sustantivo. 1 1 Un punto o lugar de cruce de un lado a otro. ÃƒÂ ¢ Ã‚â‚¬Ã‚ËœEl contratista holandés tuvo que construir una nueva carretera de circunvalación con varios tramos de... |

Mixed top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 7207104 | 0 | 0.7760 | es | El cruzamiento es el proceso de reproducción con la intención de crear descendencia que comparta los rasgos de ambos linajes parentales o para producir un animal con vigor híbri... |
| 2 | 7207097 | 1 | 0.7565 | es | El cruzamiento significa la producción de un organismo mediante el apareamiento de dos especies, razas o variedades diferentes. También puede denominarse hibridación. El cruzami... |
| 3 | 7024589 | 0 | 0.6700 | es | Cruzando. Definición. sustantivo. Un proceso que ocurre durante la meiosis en el que dos cromosomas se emparejan e intercambian segmentos de su material genético. Suplemento: es... |
| 4 | 2619772 | 0 | 0.6666 | es | El cruzamiento es un tipo de reproducción selectiva, excepto con animales de dos organismos de la misma especie pero no de la misma raza. La cría selectiva es una de las causas ... |
| 5 | 2540596 | 0 | 0.6578 | es | Mejor respuesta: cruce es otro nombre para la recombinación o el intercambio físico de partes iguales de cromátidas adyacentes no hermanas. Cuando se produce el entrecruzamiento... |
| 6 | 2540594 | 0 | 0.6551 | es | El cruce, o recombinación, es el intercambio de segmentos cromosómicos entre cromátidas no hermanas en la meiosis. El cruce crea nuevas combinaciones de genes en los gametos que... |
| 7 | 2958424 | 0 | 0.6517 | es | cruzar, 1 Biología. 2 para cambiar la lealtad, como de un partido político a otro. 3 para cambiar con éxito de un campo de actividad, género, etc., a otro: pasar del jazz al roc... |
| 8 | 4309767 | 0 | 0.6466 | es | El cruce es el intercambio de genes entre dos cromosomas, lo que da como resultado cromátidas no idénticas que comprenden el material genético de los gametos (espermatozoides y ... |
| 9 | 1342502 | 0 | 0.6432 | es | El cruce cromosómico (o cruce) es el intercambio de material genético entre cromosomas homólogos que da como resultado cromosomas recombinantes durante la reproducción sexual. |
| 10 | 2540593 | 0 | 0.6407 | es | cruzando. cruce, proceso en genética por el cual los dos cromosomas de un par homólogo intercambian segmentos iguales entre sí. El cruce ocurre en la primera división de la meio... |

### qid `1062511`

- query A (`es`): ¿Cuál es el salario de asistente administrativo de oficina en Alabama?
- query B (`zh`): 阿拉巴马州的办公室行政助理工资是多少
- diagnosis: RankDrop; nDCG@10 end=100.0000, mix=63.0930, Δ=-36.9070; Recall@10 end=100.0000, mix=100.0000, Δ=0.0000; tokens(a/b)=11/10, len_ratio=1.1000; overlap@10=9; source=evaluate_perquery

Endpoint top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 7166524 | 1 | 0.7909 | es | El salario anual de un asistente administrativo del estado de Alabama es de aproximadamente $ 36000, según los datos de la escala salarial y salarial de 13 empleados reales del ... |
| 2 | 7166526 | 0 | 0.7843 | es | 13 Sueldos de asistente administrativo del estado de Alabama. Los asistentes administrativos del estado de Alabama ganan $ 36,000 anualmente, o $ 17 por hora, que es un 9% más a... |
| 3 | 7166525 | 0 | 0.7826 | es | 13 Sueldos de asistente administrativo del estado de Alabama Examinar los sueldos del estado de Alabama por puesto de trabajo ÃƒÂ ¢ Ã‚â € Ã‚â € ™ Los asistentes administrativos ... |
| 4 | 7166528 | 0 | 0.7544 | es | El salario típico del Asistente de apoyo administrativo II del estado de Alabama es $ 25,097. Los sueldos del Asistente de Apoyo Administrativo II en el estado de Alabama pueden... |
| 5 | 7166521 | 0 | 0.7535 | es | Buscar sueldos en el estado de Alabama por puesto de trabajo ÃƒÂ ¢ Ã‚â € Ã‚â € ™. Los asistentes administrativos del estado de Alabama ganan $ 36,000 anualmente, o $ 17 por hora... |
| 6 | 7166520 | 0 | 0.7507 | es | El salario típico del Asistente de apoyo administrativo II del estado de Alabama es $ 25,097. Los sueldos del Asistente de apoyo administrativo II en el estado de Alabama pueden... |
| 7 | 107609 | 0 | 0.7399 | es | Sueldos de asistente legal en Alabama. En 2011, había 2.720 asistentes legales empleados en Alabama, la mayoría de los cuales se encontraron trabajando en el área metropolitana ... |
| 8 | 4851890 | 0 | 0.7340 | es | Sueldos de asistente legal en Alabama. En 2011, había 2.720 asistentes legales empleados en Alabama, la mayoría de los cuales se encontraron trabajando en el área metropolitana ... |
| 9 | 7166527 | 0 | 0.7302 | es | Salario promedio del Asistente de Apoyo Administrativo II del Estado de Alabama: $ 25,097. Tendencia de los sueldos del Estado de Alabama según los sueldos publicados de forma a... |
| 10 | 6184388 | 0 | 0.7125 | es | El salario promedio de un asistente de fisioterapia en Alabama es de aproximadamente $ 45,528 por año, que es un 8% por debajo del promedio nacional. La información salarial pro... |

Mixed top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 7166526 | 0 | 0.7963 | es | 13 Sueldos de asistente administrativo del estado de Alabama. Los asistentes administrativos del estado de Alabama ganan $ 36,000 anualmente, o $ 17 por hora, que es un 9% más a... |
| 2 | 7166524 | 1 | 0.7945 | es | El salario anual de un asistente administrativo del estado de Alabama es de aproximadamente $ 36000, según los datos de la escala salarial y salarial de 13 empleados reales del ... |
| 3 | 7166525 | 0 | 0.7914 | es | 13 Sueldos de asistente administrativo del estado de Alabama Examinar los sueldos del estado de Alabama por puesto de trabajo ÃƒÂ ¢ Ã‚â € Ã‚â € ™ Los asistentes administrativos ... |
| 4 | 7166521 | 0 | 0.7649 | es | Buscar sueldos en el estado de Alabama por puesto de trabajo ÃƒÂ ¢ Ã‚â € Ã‚â € ™. Los asistentes administrativos del estado de Alabama ganan $ 36,000 anualmente, o $ 17 por hora... |
| 5 | 7166528 | 0 | 0.7605 | es | El salario típico del Asistente de apoyo administrativo II del estado de Alabama es $ 25,097. Los sueldos del Asistente de Apoyo Administrativo II en el estado de Alabama pueden... |
| 6 | 7166520 | 0 | 0.7553 | es | El salario típico del Asistente de apoyo administrativo II del estado de Alabama es $ 25,097. Los sueldos del Asistente de apoyo administrativo II en el estado de Alabama pueden... |
| 7 | 107609 | 0 | 0.7494 | es | Sueldos de asistente legal en Alabama. En 2011, había 2.720 asistentes legales empleados en Alabama, la mayoría de los cuales se encontraron trabajando en el área metropolitana ... |
| 8 | 4851890 | 0 | 0.7450 | es | Sueldos de asistente legal en Alabama. En 2011, había 2.720 asistentes legales empleados en Alabama, la mayoría de los cuales se encontraron trabajando en el área metropolitana ... |
| 9 | 7166527 | 0 | 0.7373 | es | Salario promedio del Asistente de Apoyo Administrativo II del Estado de Alabama: $ 25,097. Tendencia de los sueldos del Estado de Alabama según los sueldos publicados de forma a... |
| 10 | 8318451 | 0 | 0.7168 | es | (Estados Unidos). Un asistente administrativo gana un salario promedio de $ 14,79 por hora. El pago por este trabajo no cambia mucho según la experiencia, ya que los más experim... |

### qid `1082002`

- query A (`es`): lo que eventualmente reemplazó a la industria artesanal
- query B (`zh`): 什么最终取代了家庭手工业
- diagnosis: RankDrop; nDCG@10 end=100.0000, mix=63.0930, Δ=-36.9070; Recall@10 end=100.0000, mix=100.0000, Δ=0.0000; tokens(a/b)=8/7, len_ratio=1.1429; overlap@10=9; source=evaluate_perquery

Endpoint top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 7147611 | 1 | 0.6769 | es | La industria artesanal que fue reemplazada por los molinos fue la producción de hilados e hilos de algodón. La producción de harina también fue una industria artesanal que fue r... |
| 2 | 7787672 | 0 | 0.6589 | es | existe y es una alternativa de. Mejorando el mundo, una respuesta a la vez. La industria artesanal terminó por el hecho de que la gente se estaba mudando a la ciudad donde había... |
| 3 | 7787674 | 0 | 0.6439 | es | Respondido por la comunidad WikiAnswersÃƒâ € šÃ‚Â®. Mejorando el mundo, una respuesta a la vez. La industria artesanal terminó por el hecho de que la gente se estaba mudando a l... |
| 4 | 2570413 | 0 | 0.6271 | zh | 在工业革命期间，商品的生产方式发生了变化。机器开始帮助并最终取代工匠，而不是利用工匠来生产手工制品。 |
| 5 | 7787669 | 0 | 0.5911 | es | La revolución industrial introdujo una amplia gama de productos que podían producirse en masa y distribuirse de forma económica. El auge de la revolución industrial significó el... |
| 6 | 2805528 | 0 | 0.5812 | es | Durante la Revolución Industrial del siglo XIX, las máquinas se hicieron cargo de la mayor parte del trabajo de fabricación de los hombres y las fábricas reemplazaron los taller... |
| 7 | 4923688 | 0 | 0.5748 | zh | 四年后，当战争结束时，风格发生了变化，生产方式也发生了革命性的变化……因此，工业革命发生了。手工制作的一种物品，例如铁床，不再具有成本效益，并已成为过去。 |
| 8 | 4738387 | 0 | 0.5700 | es | Cuando la idea de la industria artesanal surgió inicialmente a fines del siglo XVI, la mayoría de los fabricantes producían servicios basados ​​en textiles como costura, confecc... |
| 9 | 2438659 | 0 | 0.5673 | es | Avances como estos fueron evidentes en todas las industrias durante esta era. Durante la Revolución Industrial se produjeron cambios en la forma en que se producían los bienes. ... |
| 10 | 8151286 | 0 | 0.5669 | zh | 到 19 世纪末，纺织厂和其他工厂生产了种类繁多的新产品，并产生了大量新的支持行业、金融机构以及交通和信息网络。古老的工匠和农业生活方式已经消失。 |

Mixed top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 7787672 | 0 | 0.6704 | es | existe y es una alternativa de. Mejorando el mundo, una respuesta a la vez. La industria artesanal terminó por el hecho de que la gente se estaba mudando a la ciudad donde había... |
| 2 | 7147611 | 1 | 0.6701 | es | La industria artesanal que fue reemplazada por los molinos fue la producción de hilados e hilos de algodón. La producción de harina también fue una industria artesanal que fue r... |
| 3 | 2570413 | 0 | 0.6668 | zh | 在工业革命期间，商品的生产方式发生了变化。机器开始帮助并最终取代工匠，而不是利用工匠来生产手工制品。 |
| 4 | 7787674 | 0 | 0.6581 | es | Respondido por la comunidad WikiAnswersÃƒâ € šÃ‚Â®. Mejorando el mundo, una respuesta a la vez. La industria artesanal terminó por el hecho de que la gente se estaba mudando a l... |
| 5 | 4923688 | 0 | 0.6155 | zh | 四年后，当战争结束时，风格发生了变化，生产方式也发生了革命性的变化……因此，工业革命发生了。手工制作的一种物品，例如铁床，不再具有成本效益，并已成为过去。 |
| 6 | 2805528 | 0 | 0.6108 | zh | 在 19 世纪的工业革命期间，机器接管了大部分制造工作，工厂取代了工匠的作坊。 |
| 7 | 2438659 | 0 | 0.6055 | zh | 在这个时代，这些进步在所有行业中都很明显。在工业革命期间，商品的生产方式发生了变化。机器开始帮助并最终取代工匠，而不是利用工匠来生产手工制品。 |
| 8 | 7787669 | 0 | 0.6000 | es | La revolución industrial introdujo una amplia gama de productos que podían producirse en masa y distribuirse de forma económica. El auge de la revolución industrial significó el... |
| 9 | 7922453 | 0 | 0.5971 | zh | 工业革命见证了工厂的诞生，工厂开始生产许多农村人手工制作的商品，工厂使它们更便宜并用于大众消费，尤其是纺织品。传统上在家中使用织机编织纺织品的人无法竞争。 |
| 10 | 8151286 | 0 | 0.5912 | zh | 到 19 世纪末，纺织厂和其他工厂生产了种类繁多的新产品，并产生了大量新的支持行业、金融机构以及交通和信息网络。古老的工匠和农业生活方式已经消失。 |

### qid `1082603`

- query A (`es`): ¿Para qué sirve el examinador de visión suresight?
- query B (`zh`): Suresight 视力筛查仪的屏幕内容是什么？
- diagnosis: RankDrop; nDCG@10 end=100.0000, mix=63.0930, Δ=-36.9070; Recall@10 end=100.0000, mix=100.0000, Δ=0.0000; tokens(a/b)=8/8, len_ratio=1.0000; overlap@10=9; source=evaluate_perquery

Endpoint top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 7144202 | 1 | 0.6356 | es | El desglose: los profesionales de la salud utilizan el SureSight Vision Screener para la detección de la agudeza visual, y debe informarlo con 99173 (Prueba de detección de la a... |
| 2 | 7144206 | 0 | 0.6324 | es | Apto para niños: las luces y los sonidos intermitentes de SureSight Vision Screener atraen la atención del niño. Rápido y eficiente: la prueba automática de cinco segundos resue... |
| 3 | 7144204 | 0 | 0.6261 | es | La tecnología de detección más nueva que utiliza una máquina, como SureSight Vision Screener, parece superar estos desafíos al proporcionar resultados más rápido con un método m... |
| 4 | 7144199 | 0 | 0.6247 | zh | SureSightÃƒâ€šÃ‚Â® Vision Screener。快速概览。产品状态：仅限服务。要查看可用的零件和附件，请单击零件和附件选项卡。此产品不再销售。要查看支持选项、零件和附件，请单击进入产品页面。 |
| 5 | 7144205 | 0 | 0.6086 | zh | 在购买用于视力筛查的 SureSight 之前了解这些事实。 - 发布于 2007 年 11 月 29 日星期四。 u 注意：产品信息可能会将您误导到 92015。如果您正在考虑用新技术设备替换技术上困难且耗时的 Snellen 视力表测试，权衡这些利弊。 |
| 6 | 7144201 | 0 | 0.6068 | es | Beneficios del evaluador de visión SureSight 140 de Welch Allyn. Compacto y portátil: la unidad portátil funciona durante tres horas de prueba continua. Diseño cómodo y ergonómi... |
| 7 | 6537778 | 0 | 0.6055 | es | Está diseñado para ayudar a los conductores, no para reemplazarlos. EyeSight es la nueva tecnología de Subaru que monitorea su seguridad, ya sea que sus ojos estén en la carrete... |
| 8 | 6537779 | 0 | 0.5969 | es | Subaru EyeSight es una tecnología de asistencia al conductor desarrollada por los ingenieros de Subaru como un medio para ayudar a que los vehículos Subaru se encuentren entre l... |
| 9 | 4363490 | 0 | 0.5942 | zh | EyeSight 是一种驾驶辅助系统，它使用一系列功能来帮助驾驶员做出决策，以提供更安全和舒适的驾驶并减少驾驶员疲劳。它旨在帮助驾驶员，而不是取代他们。 |
| 10 | 67279 | 0 | 0.5908 | es | Los exámenes de la vista pueden ser un paso clave para romper el ciclo de la pobreza. Ya sea en el extranjero o aquí en casa, optometristas independientes voluntarios en LensCra... |

Mixed top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 7144206 | 0 | 0.6733 | zh | 儿童友好：SureSight 视力筛查仪的闪烁灯光和声音吸引孩子的注意力。快速高效：五秒钟的自动测试解决了合规性问题并使测试更容易——不再需要五分钟的视力表检查！ |
| 2 | 7144202 | 1 | 0.6597 | es | El desglose: los profesionales de la salud utilizan el SureSight Vision Screener para la detección de la agudeza visual, y debe informarlo con 99173 (Prueba de detección de la a... |
| 3 | 7144199 | 0 | 0.6571 | zh | SureSightÃƒâ€šÃ‚Â® Vision Screener。快速概览。产品状态：仅限服务。要查看可用的零件和附件，请单击零件和附件选项卡。此产品不再销售。要查看支持选项、零件和附件，请单击进入产品页面。 |
| 4 | 7144205 | 0 | 0.6556 | zh | 在购买用于视力筛查的 SureSight 之前了解这些事实。 - 发布于 2007 年 11 月 29 日星期四。 u 注意：产品信息可能会将您误导到 92015。如果您正在考虑用新技术设备替换技术上困难且耗时的 Snellen 视力表测试，权衡这些利弊。 |
| 5 | 7144204 | 0 | 0.6547 | es | La tecnología de detección más nueva que utiliza una máquina, como SureSight Vision Screener, parece superar estos desafíos al proporcionar resultados más rápido con un método m... |
| 6 | 7144201 | 0 | 0.6286 | es | Beneficios del evaluador de visión SureSight 140 de Welch Allyn. Compacto y portátil: la unidad portátil funciona durante tres horas de prueba continua. Diseño cómodo y ergonómi... |
| 7 | 7144198 | 0 | 0.6177 | zh | Welch Allyn SureSight 140 视力筛查仪 (#14000) 包括一个紧凑型充电器和存储支架。它经过认证，符合正确的操作规范并且状况良好。保修：1 年 - 零件和人工。 |
| 8 | 6537778 | 0 | 0.6169 | es | Está diseñado para ayudar a los conductores, no para reemplazarlos. EyeSight es la nueva tecnología de Subaru que monitorea su seguridad, ya sea que sus ojos estén en la carrete... |
| 9 | 4363490 | 0 | 0.6118 | zh | EyeSight 是一种驾驶辅助系统，它使用一系列功能来帮助驾驶员做出决策，以提供更安全和舒适的驾驶并减少驾驶员疲劳。它旨在帮助驾驶员，而不是取代他们。 |
| 10 | 6537779 | 0 | 0.6113 | es | Subaru EyeSight es una tecnología de asistencia al conductor desarrollada por los ingenieros de Subaru como un medio para ayudar a que los vehículos Subaru se encuentren entre l... |

