# Case Report: CASE_0246

## 1) Case Header

- pair: `EN-VI`
- doc_mix: `VI docs`
- model: `bge-m3`
- method: `embed`
- doc_index_id: `mmarco-8841823-vietnamese-en-vi-5bands-bge-m3`
- endpoint lambda: `1.0`
- lambda*: `0.5`
- overall delta (mixed - endpoint): `2.2578999999999994` (CI90: [1.5192888973265315, 3.0766107055306784])

## 2) How Many Queries Drive the Gain

- metric source counts: evaluate_perquery=1484, recomputed_from_run_qrels=0
- ΔnDCG@10 quantiles (all queries): min=-100.0000, p25=0.0000, p50=0.0000, p75=0.0000, max=100.0000
- best-100 mean ΔnDCG@10: `56.0101`
- control-20 mean ΔnDCG@10: `0.0000`

## 3) Failure Label Breakdown (Best Set)

- label thresholds: mismatch_rate_mix>0.0000, endpoint_cos<0.5000, len_ratio<0.5000 or >1.5000, delta_recall<0.0000, rankdrop=(delta_ndcg<0.0000 and delta_recall>=0.0000)
- IndexLeakage: count=0, mean ΔnDCG@10=
- TranslationDivergence: count=6, mean ΔnDCG@10=68.7202
- RecallDrop: count=0, mean ΔnDCG@10=
- RankDrop: count=0, mean ΔnDCG@10=
- Unclassified: count=94, mean ΔnDCG@10=55.1988

## 4) Top 20 Best Queries

| qid | metric_source | ndcg_end | ndcg_mix | d_ndcg | rec_end | rec_mix | d_rec | first_end | first_mix | rank_shift | ov10 | ov50 | tok_a | tok_b | len_ratio | endpoint_cos | r | delta_perp | cos_to_a | cos_to_b | mismatch_end | mismatch_mix | ascii_end | ascii_mix | label |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1085327 | evaluate_perquery | 0.0000 | 100.0000 | 100.0000 | 0.0000 | 100.0000 | 100.0000 | inf | 1 | -9998.0000 | 1 | 6 | 6 | 6 | 1.0000 | 0.3978 | 0.5000 | 0.0000 | 0.8360 | 0.8360 | 0.0000 | 0.0000 | 0.7582 | 0.7842 | TranslationDivergence |
| 1086594 | evaluate_perquery | 0.0000 | 100.0000 | 100.0000 | 0.0000 | 100.0000 | 100.0000 | 12 | 1 | -11.0000 | 5 | 32 | 9 | 16 | 0.5625 | 0.6128 | 0.5000 | 0.0000 | 0.8980 | 0.8980 | 0.0000 | 0.0000 | 0.7592 | 0.7515 | Unclassified |
| 1095121 | evaluate_perquery | 0.0000 | 100.0000 | 100.0000 | 0.0000 | 100.0000 | 100.0000 | inf | 1 | -9998.0000 | 6 | 23 | 10 | 12 | 0.8333 | 0.6906 | 0.5000 | 0.0000 | 0.9194 | 0.9194 | 0.0000 | 0.0000 | 0.7155 | 0.7287 | Unclassified |
| 1095874 | evaluate_perquery | 0.0000 | 100.0000 | 100.0000 | 0.0000 | 100.0000 | 100.0000 | 47 | 1 | -46.0000 | 4 | 36 | 6 | 11 | 0.5455 | 0.8109 | 0.5000 | 0.0000 | 0.9515 | 0.9515 | 0.0000 | 0.0000 | 0.7069 | 0.7324 | Unclassified |
| 1097223 | evaluate_perquery | 0.0000 | 100.0000 | 100.0000 | 0.0000 | 100.0000 | 100.0000 | inf | 1 | -9998.0000 | 0 | 7 | 7 | 10 | 0.7000 | 0.6111 | 0.5000 | 0.0000 | 0.8975 | 0.8975 | 0.0000 | 0.0000 | 0.6635 | 0.7232 | Unclassified |
| 230725 | evaluate_perquery | 0.0000 | 100.0000 | 100.0000 | 0.0000 | 100.0000 | 100.0000 | inf | 1 | -9998.0000 | 3 | 24 | 7 | 6 | 1.1667 | 0.6290 | 0.5000 | 0.0000 | 0.9025 | 0.9025 | 0.0000 | 0.0000 | 0.7510 | 0.7503 | Unclassified |
| 249821 | evaluate_perquery | 0.0000 | 100.0000 | 100.0000 | 0.0000 | 100.0000 | 100.0000 | inf | 1 | -9998.0000 | 2 | 27 | 9 | 5 | 1.8000 | 0.5705 | 0.5000 | 0.0000 | 0.8862 | 0.8862 | 0.0000 | 0.0000 | 0.7052 | 0.7069 | TranslationDivergence |
| 277701 | evaluate_perquery | 0.0000 | 100.0000 | 100.0000 | 0.0000 | 100.0000 | 100.0000 | inf | 1 | -9998.0000 | 4 | 30 | 8 | 11 | 0.7273 | 0.7649 | 0.5000 | 0.0000 | 0.9394 | 0.9394 | 0.0000 | 0.0000 | 0.7596 | 0.7925 | Unclassified |
| 288702 | evaluate_perquery | 0.0000 | 100.0000 | 100.0000 | 0.0000 | 100.0000 | 100.0000 | inf | 1 | -9998.0000 | 7 | 35 | 8 | 12 | 0.6667 | 0.8028 | 0.5000 | 0.0000 | 0.9494 | 0.9494 | 0.0000 | 0.0000 | 0.7472 | 0.7465 | Unclassified |
| 299094 | evaluate_perquery | 0.0000 | 100.0000 | 100.0000 | 0.0000 | 100.0000 | 100.0000 | 15 | 1 | -14.0000 | 4 | 15 | 6 | 9 | 0.6667 | 0.5186 | 0.5000 | 0.0000 | 0.8714 | 0.8714 | 0.0000 | 0.0000 | 0.6745 | 0.6835 | Unclassified |
| 412319 | evaluate_perquery | 0.0000 | 100.0000 | 100.0000 | 0.0000 | 100.0000 | 100.0000 | 19 | 1 | -18.0000 | 5 | 23 | 8 | 11 | 0.7273 | 0.6661 | 0.5000 | 0.0000 | 0.9127 | 0.9127 | 0.0000 | 0.0000 | 0.7316 | 0.7400 | Unclassified |
| 424408 | evaluate_perquery | 0.0000 | 100.0000 | 100.0000 | 0.0000 | 100.0000 | 100.0000 | 15 | 1 | -14.0000 | 2 | 23 | 7 | 11 | 0.6364 | 0.7653 | 0.5000 | 0.0000 | 0.9395 | 0.9395 | 0.0000 | 0.0000 | 0.8092 | 0.8138 | Unclassified |
| 880766 | evaluate_perquery | 0.0000 | 100.0000 | 100.0000 | 0.0000 | 100.0000 | 100.0000 | inf | 1 | -9998.0000 | 0 | 5 | 6 | 5 | 1.2000 | 0.5863 | 0.5000 | 0.0000 | 0.8906 | 0.8906 | 0.0000 | 0.0000 | 0.7002 | 0.7465 | Unclassified |
| 927196 | evaluate_perquery | 0.0000 | 100.0000 | 100.0000 | 0.0000 | 100.0000 | 100.0000 | inf | 1 | -9998.0000 | 0 | 18 | 12 | 16 | 0.7500 | 0.7557 | 0.5000 | 0.0000 | 0.9369 | 0.9369 | 0.0000 | 0.0000 | 0.6973 | 0.7228 | Unclassified |
| 333327 | evaluate_perquery | 30.1030 | 100.0000 | 69.8970 | 100.0000 | 100.0000 | 0.0000 | 9 | 1 | -8.0000 | 5 | 35 | 11 | 13 | 0.8462 | 0.6891 | 0.5000 | 0.0000 | 0.9190 | 0.9190 | 0.0000 | 0.0000 | 0.6882 | 0.7093 | Unclassified |
| 1099670 | evaluate_perquery | 31.5465 | 100.0000 | 68.4535 | 100.0000 | 100.0000 | 0.0000 | 8 | 1 | -7.0000 | 7 | 25 | 6 | 8 | 0.7500 | 0.6215 | 0.5000 | 0.0000 | 0.9004 | 0.9004 | 0.0000 | 0.0000 | 0.7298 | 0.7371 | Unclassified |
| 264594 | evaluate_perquery | 31.5465 | 100.0000 | 68.4535 | 100.0000 | 100.0000 | 0.0000 | 8 | 1 | -7.0000 | 6 | 21 | 6 | 8 | 0.7500 | 0.7225 | 0.5000 | 0.0000 | 0.9280 | 0.9280 | 0.0000 | 0.0000 | 0.7105 | 0.7158 | Unclassified |
| 577167 | evaluate_perquery | 33.3333 | 100.0000 | 66.6667 | 100.0000 | 100.0000 | 0.0000 | 7 | 1 | -6.0000 | 7 | 34 | 9 | 7 | 1.2857 | 0.8581 | 0.5000 | 0.0000 | 0.9639 | 0.9639 | 0.0000 | 0.0000 | 0.7128 | 0.7160 | Unclassified |
| 988306 | evaluate_perquery | 33.3333 | 100.0000 | 66.6667 | 100.0000 | 100.0000 | 0.0000 | 7 | 1 | -6.0000 | 9 | 46 | 7 | 10 | 0.7000 | 0.9062 | 0.5000 | 0.0000 | 0.9763 | 0.9763 | 0.0000 | 0.0000 | 0.7796 | 0.7840 | Unclassified |
| 1051755 | evaluate_perquery | 35.6207 | 100.0000 | 64.3793 | 100.0000 | 100.0000 | 0.0000 | 6 | 1 | -5.0000 | 7 | 29 | 7 | 10 | 0.7000 | 0.7771 | 0.5000 | 0.0000 | 0.9426 | 0.9426 | 0.0000 | 0.0000 | 0.7370 | 0.7230 | Unclassified |

## 5) Per-Query Diff Blocks (Top 20 Best)

All metric deltas are `mixed - endpoint` in 0-100 point units.

Note: `retrieval_score_raw` below is the original run ranking score from `.trec`, not an evaluation metric and not on the 0-100 nDCG/Recall scale.

### qid `1085327`

- query A (`en`): what county is ridgway pa in
- query B (`vi`): đường đua nằm ở quận nào
- diagnosis: TranslationDivergence; nDCG@10 end=0.0000, mix=100.0000, Δ=100.0000; Recall@10 end=0.0000, mix=100.0000, Δ=100.0000; tokens(a/b)=6/6, len_ratio=1.0000; overlap@10=1; source=evaluate_perquery; focus=best (gain)

Endpoint top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 4211066 | 0 | 0.6288 | vi | Cuộc đua diễn ra trên các con đường của trung tâm thành phố Long Beach bao quanh Trung tâm Hội nghị và Giải trí Long Beach. Vạch xuất phát / kết thúc nằm trên Shoreline Drive. K... |
| 2 | 5086664 | 0 | 0.6279 | vi | Đường đua Aqueduct Đường đua Aqueduct, còn được gọi là Big A, là đường đua duy nhất ở Thành phố New York, chiếm 210 mẫu Anh trong Công viên Nam Ozone ở quận Queens. Trong ba năm... |
| 3 | 5086667 | 0 | 0.6242 | vi | Đường đua Aqueduct. Đường đua Aqueduct, còn được gọi là Big A, là đường đua duy nhất ở Thành phố New York, chiếm 210 mẫu Anh trong Công viên Nam Ozone ở quận Queens. Trong ba nă... |
| 4 | 3310335 | 0 | 0.6140 | vi | Công viên Đường đua Pittsburgh. Công viên Đường đua Pittsburgh là một con đường kéo dài 1/4 dặm nằm gần New Alexandria, Pennsylvania. Đường ray này dường như đang sử dụng một đư... |
| 5 | 3915952 | 0 | 0.5964 | vi | Công viên Ozone là một khu phố của Thành phố New York nằm ở quận Queens giáp với Woodhaven, Richmond Hill, Bãi biển Howard và quận Brooklyn. Đây là quê hương của Đường đua Aqued... |
| 6 | 5086672 | 0 | 0.5945 | vi | Đường đua Aqueduct. Đường đua 1 Aqueduct, còn được gọi là Big A, là đường đua duy nhất ở Thành phố New York, chiếm 210 mẫu Anh trong Công viên Nam Ozone ở quận Queens. 2 Một tro... |
| 7 | 948070 | 0 | 0.5938 | vi | Không phải tên chính xác như vậy, nhưng có những đường đua khác trong khu vực Los Angeles. Trong khi thiết kế bên ngoài của mặt tiền của đường đua dựa nhiều vào Rose Bowl ở Pasa... |
| 8 | 1849320 | 0 | 0.5903 | vi | Đấu trường tọa lạc tại số 1280 Bollenbacher Dr., ngay gần Quốc lộ 3 về phía Nam về phía Tây. Đi về phía nam, rẽ phải trên Jefferson Parkway, đi một khối và rẽ trái vào Bollenbac... |
| 9 | 2958142 | 0 | 0.5884 | vi | Cuộc diễu hành sẽ bắt đầu trên Broadway tại Đường 11, rẽ phải trên Đại lộ Grand, và phải trên Harrison đến Lakeside Drive, kết thúc ở Oak và tiến đến Trung tâm Hội nghị Henry J.... |
| 10 | 8804225 | 0 | 0.5840 | vi | Nó nằm ở phía đông nam của quận trung tâm thành phố. Nó thường được coi là khu vực giáp với Đại lộ Hollywood ở phía Bắc với Đường 85 ở phía Nam; từ Đại lộ Line ở phía Đông đến Đ... |

Mixed top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 7119970 | 1 | 0.6376 | vi | Ridgway là một quận trong và quận lỵ của Quận Elk, Pennsylvania, Hoa Kỳ. Ridgway được thành lập bởi thương gia vận tải biển Jacob Ridgway và James Gillis của Philadelphia. Jacob... |
| 2 | 2190708 | 0 | 0.5965 | vi | Thị trấn Ridgeway, nằm ở Quận Iowa, Wisconsin bao gồm 48,75 dặm đường dọc theo hàng dặm với phong cảnh tráng lệ xuyên qua rừng sâu, khung cảnh rộng lớn từ những rặng núi cao và ... |
| 3 | 238691 | 0 | 0.5890 | vi | Midway, Quận Washington, Pennsylvania. Midway là một quận ở Quận Washington thuộc tiểu bang Pennsylvania của Hoa Kỳ. Dân số tại thời điểm điều tra dân số năm 2000 là 982 người. ... |
| 4 | 227713 | 0 | 0.5849 | vi | Công viên Ridley là một quận ở Quận Delaware, Pennsylvania, Hoa Kỳ. Dân số tại thời điểm điều tra dân số năm 2010 là 7.002 người. Công viên Ridley là nhà của bộ phận trực thăng ... |
| 5 | 2190706 | 0 | 0.5775 | vi | Ridgeway là một ngôi làng ở Iowa County, Wisconsin, Hoa Kỳ. Đây là cộng đồng đông dân thứ tư ở Iowa County. Dân số tại thời điểm điều tra dân số năm 2010 là 653 người. Ngôi làng... |
| 6 | 3310335 | 0 | 0.5702 | vi | Công viên Đường đua Pittsburgh. Công viên Đường đua Pittsburgh là một con đường kéo dài 1/4 dặm nằm gần New Alexandria, Pennsylvania. Đường ray này dường như đang sử dụng một đư... |
| 7 | 2322072 | 0 | 0.5628 | vi | Cơ quan Khảo sát Địa chất Hoa Kỳ Hệ thống Thông tin Tên Địa lý: 1174013. Edgewood là một quận ở Quận Allegheny, Pennsylvania, Hoa Kỳ, tiếp giáp với thành phố Pittsburgh. Dân số ... |
| 8 | 2474855 | 0 | 0.5609 | vi | Martin nằm ở phía đông nam Quận Allegan, cách giao lộ US-131 và M-222 (M-118 cũ) một dặm về phía đông. Nó là một cộng đồng nông nghiệp nhỏ bao gồm một số cửa hàng và doanh nghiệ... |
| 9 | 1524545 | 0 | 0.5591 | vi | Phía tây bắc giáp với quận Swarthmore, phía bắc giáp Springfield Township và quận Rutledge, phía đông giáp đường 420 Pennsylvania, phía đông nam giáp quận Prospect Park, phía na... |
| 10 | 2837263 | 0 | 0.5586 | vi | Các hố trong cuộc đua NASCAR Cup năm 1985. Richmond International Raceway (RIR) là một đường đua trải nhựa hình chữ D dài 3/4 dặm (1,2 km) nằm ngay bên ngoài Richmond, Virginia ... |

### qid `1086594`

- query A (`en`): what are valence electrons used by an element worksheet
- query B (`vi`): các electron hóa trị được sử dụng bởi một bảng tính nguyên tố là gì
- diagnosis: Unclassified; nDCG@10 end=0.0000, mix=100.0000, Δ=100.0000; Recall@10 end=0.0000, mix=100.0000, Δ=100.0000; tokens(a/b)=9/16, len_ratio=0.5625; overlap@10=5; source=evaluate_perquery; focus=best (gain)

Endpoint top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 4994953 | 0 | 0.7304 | vi | Sử dụng bảng tuần hoàn. Đối với các nguyên tố nằm trong cột A, số electron hóa trị bằng số thứ tự của cột: IA: 1 electron hóa trị (liti, v.v.) IIA: 2 electron hóa trị (beri, v.v... |
| 2 | 3424860 | 0 | 0.7259 | vi | Các nguyên tố trong một nhóm của bảng tuần hoàn có cùng số electron hóa trị trong cùng hình dạng của các obitan. Nói chung, số electron hóa trị của nguyên tử bằng số nhóm của nó... |
| 3 | 7108824 | 0 | 0.7223 | vi | Tên: _____ Tiết: __ Electron và Ion Hóa trị Bảng tính Các electron hóa trị 1. Electron hóa trị là gì và tại sao chúng lại quan trọng đối với một nhà hóa học? 2. Xu hướng tuần ho... |
| 4 | 4315721 | 0 | 0.7200 | vi | Các nguyên tố trong một nhóm của bảng tuần hoàn có cùng số electron hóa trị trong cùng hình dạng của các obitan. Nói chung, số electron hóa trị của nguyên tử bằng số nhóm của nó... |
| 5 | 3424853 | 0 | 0.7182 | vi | Trong Hóa học, các electron hóa trị là các electron nằm ở lớp vỏ electron ngoài cùng của nguyên tố. Biết cách tìm số electron hóa trị trong một nguyên tử cụ thể là một kỹ năng q... |
| 6 | 8398881 | 0 | 0.7182 | vi | Ngoài hydro, tất cả các nguyên tố trong nhóm I của bảng tuần hoàn đều có một electron hóa trị. Chúng là liti, natri, kali, rubidi, xêzi và franxi. Fa ÃƒÂ ¢ Ã‚â‚¬Ã‚Â¦ bạn di chuy... |
| 7 | 8149963 | 0 | 0.7180 | vi | Các electron hóa trị là các electron ở lớp vỏ electron ngoài cùng của nguyên tử cô lập của một nguyên tố. Đôi khi, nó cũng được coi là cơ sở của Bảng tuần hoàn hiện đại. Trong m... |
| 8 | 8398884 | 0 | 0.7163 | vi | Ngoài hydro, tất cả các nguyên tố trong nhóm I của bảng tuần hoàn đều có một electron hóa trị. Chúng là liti, natri, kali, rubidi, xêzi và franxi. ÃƒÂ ¢ Ã‚â‚¬Ã‚Â¦ bạn di chuyển ... |
| 9 | 5435089 | 0 | 0.7158 | vi | Bảng tuần hoàn được thiết kế với tính năng này. Mỗi nguyên tố có số electron hóa trị bằng số thứ tự nhóm của nó trên Bảng tuần hoàn. Bảng này minh họa một số đặc điểm thú vị và ... |
| 10 | 8398886 | 0 | 0.7127 | vi | Làm cho thế giới tốt đẹp hơn, một câu trả lời tại một thời điểm. Ngoài hydro, tất cả các nguyên tố trong nhóm I của bảng tuần hoàn đều có một electron hóa trị. Chúng là liti, na... |

Mixed top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 7108821 | 1 | 0.6982 | vi | Các điện tử hóa trị được sử dụng trong liên kết và xác định tính chất / đặc điểm của các nguyên tố. Các bài về electron hóa trị được viết bởi Liz LaRosa. that Element! ÃƒÂ ¢ Ã‚â... |
| 2 | 7491456 | 0 | 0.6839 | vi | Câu trả lời hay nhất: Các electron hóa trị được tìm thấy bằng cách tìm cột nào (trong số 8 cao nhất) của nguyên tố trong bảng tuần hoàn. Neon nằm ở cột thứ 8 nên nó có 8 electro... |
| 3 | 4994953 | 0 | 0.6684 | vi | Sử dụng bảng tuần hoàn. Đối với các nguyên tố nằm trong cột A, số electron hóa trị bằng số thứ tự của cột: IA: 1 electron hóa trị (liti, v.v.) IIA: 2 electron hóa trị (beri, v.v... |
| 4 | 5435089 | 0 | 0.6579 | vi | Bảng tuần hoàn được thiết kế với tính năng này. Mỗi nguyên tố có số electron hóa trị bằng số thứ tự nhóm của nó trên Bảng tuần hoàn. Bảng này minh họa một số đặc điểm thú vị và ... |
| 5 | 7108824 | 0 | 0.6572 | vi | Tên: _____ Tiết: __ Electron và Ion Hóa trị Bảng tính Các electron hóa trị 1. Electron hóa trị là gì và tại sao chúng lại quan trọng đối với một nhà hóa học? 2. Xu hướng tuần ho... |
| 6 | 8398881 | 0 | 0.6546 | vi | Ngoài hydro, tất cả các nguyên tố trong nhóm I của bảng tuần hoàn đều có một electron hóa trị. Chúng là liti, natri, kali, rubidi, xêzi và franxi. Fa ÃƒÂ ¢ Ã‚â‚¬Ã‚Â¦ bạn di chuy... |
| 7 | 5311255 | 0 | 0.6538 | vi | Biểu đồ chấm electron là một phương pháp viết ký hiệu hóa học của một nguyên tố bằng cách bao quanh nó bằng các dấu chấm để biểu thị số electron hóa trị. Các electron hóa trị đư... |
| 8 | 7108820 | 0 | 0.6527 | vi | của các electron hóa trị mà nguyên tố của bạn có. ÃƒÂ ¢ Ã‚â‚¬Ã‚Â ¢ Bạn sẽ chỉ vẽ các electron hóa trị. |
| 9 | 8398884 | 0 | 0.6517 | vi | Ngoài hydro, tất cả các nguyên tố trong nhóm I của bảng tuần hoàn đều có một electron hóa trị. Chúng là liti, natri, kali, rubidi, xêzi và franxi. ÃƒÂ ¢ Ã‚â‚¬Ã‚Â¦ bạn di chuyển ... |
| 10 | 8841543 | 0 | 0.6516 | vi | Sử dụng bảng tuần hoàn sau để trả lời các câu hỏi tiếp theo: Tìm số electron hóa trị trong các nguyên tố sau. a) Khí oxi. b) Rađon. c) Boron. Phần tử nào được biểu diễn bằng Biể... |

### qid `1095121`

- query A (`en`): how old to be to work at buffalo wild wings
- query B (`vi`): bao nhiêu tuổi để làm việc ở cánh đồng hoang dã trâu
- diagnosis: Unclassified; nDCG@10 end=0.0000, mix=100.0000, Δ=100.0000; Recall@10 end=0.0000, mix=100.0000, Δ=100.0000; tokens(a/b)=10/12, len_ratio=0.8333; overlap@10=6; source=evaluate_perquery; focus=best (gain)

Endpoint top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 5853955 | 0 | 0.6251 | vi | Tuy nhiên, caddie phải từ 14 tuổi trở lên. Người trông trẻ không cần phải có giấy chứng nhận lao động. Tuy nhiên, người trông trẻ phải từ 14 tuổi trở lên. Trẻ vị thành niên 16 v... |
| 2 | 1406363 | 0 | 0.6236 | vi | Yêu cầu về độ tuổi tối thiểu / tối đa: Hầu hết các sở động vật hoang dã của tiểu bang yêu cầu ứng viên phải từ 21 tuổi trở lên, với tất cả các sở yêu cầu ứng viên phải từ 18 tuổ... |
| 3 | 7354877 | 0 | 0.6196 | vi | Trẻ vị thành niên dưới 14 tuổi được phép làm việc trong các trang trại nhưng phải do nông dân thuê. Trẻ em có thể làm người đưa tin khi 11 tuổi và làm caddie khi 12 tuổi. |
| 4 | 6669973 | 0 | 0.6133 | vi | Thanh niên dưới 12 tuổi chỉ có thể làm việc trong lĩnh vực nông nghiệp trong trang trại nếu trang trại đó không phải trả mức lương tối thiểu của Liên bang. Thanh niên từ 16 tuổi... |
| 5 | 8293902 | 0 | 0.6127 | vi | Bạn phải bao nhiêu tuổi để làm việc tiết kiệm? |
| 6 | 7613991 | 0 | 0.6105 | vi | Xem tất cả Trâu rừng nướng cánh rừng và quán bar Việc làm tối thiểu 16 tuổi Không tìm được việc làm bạn đang tìm? Thể hiện sự quan tâm bằng cách cho chúng tôi biết bạn muốn làm ... |
| 7 | 866059 | 0 | 0.6084 | vi | A: Bạn phải đủ 18 tuổi để làm việc tại Woodward Camp. Nếu bạn quan tâm đến một vị trí, vui lòng liên hệ với văn phòng trại để biết thêm chi tiết hoặc xem phần Trang web về các c... |
| 8 | 5192868 | 0 | 0.6077 | vi | Trả lời Độ tuổi tối thiểu để làm việc khác nhau tùy theo tình huống. Ở Hoa Kỳ, độ tuổi tối thiểu cho hầu hết công việc là 16, mặc dù trong một số trường hợp, nó có thể thấp tới ... |
| 9 | 944558 | 0 | 0.6054 | vi | Giờ được quy định nếu bạn dưới 16 tuổi. Trẻ em 12 và 13 chỉ có thể làm việc trong các trang trại và được sự đồng ý của cha mẹ. Trẻ em 14 và 15 có thể có những công việc chính th... |
| 10 | 5829982 | 0 | 0.5998 | vi | Để làm việc cho NATS, bạn phải từ 13 tuổi trở lên vào thời điểm mùa giải bắt đầu. (Chúng tôi nói ngày 1 tháng 7 vì sự đơn giảnÃƒÂ ¢ Ã‚â‚¬Ã‚â „¢ s sake). Mặc dù độ tuổi hợp pháp ... |

Mixed top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 7613985 | 1 | 0.7217 | vi | Bạn phải bao nhiêu tuổi để làm việc tại Buffalo Wild Wings? Ứng viên phải từ 16 tuổi trở lên mới được làm việc tại nhà hàng này. Ứng dụng có thể in cho Buffalo Wild Wings không ... |
| 2 | 1406363 | 0 | 0.6777 | vi | Yêu cầu về độ tuổi tối thiểu / tối đa: Hầu hết các sở động vật hoang dã của tiểu bang yêu cầu ứng viên phải từ 21 tuổi trở lên, với tất cả các sở yêu cầu ứng viên phải từ 18 tuổ... |
| 3 | 866059 | 0 | 0.6617 | vi | A: Bạn phải đủ 18 tuổi để làm việc tại Woodward Camp. Nếu bạn quan tâm đến một vị trí, vui lòng liên hệ với văn phòng trại để biết thêm chi tiết hoặc xem phần Trang web về các c... |
| 4 | 8293902 | 0 | 0.6595 | vi | Bạn phải bao nhiêu tuổi để làm việc tiết kiệm? |
| 5 | 7613988 | 0 | 0.6416 | vi | 846 Buffalo Wild Wings Grill And Bar Tuổi tối thiểu 16 tuổi đang tuyển dụng việc làm gần bạn. Tìm việc làm Buffalo Wild Wings Grill And Bar Tuổi tối thiểu 16 tuổi và nộp đơn trự... |
| 6 | 5891949 | 0 | 0.6398 | vi | Độ tuổi hợp pháp để làm việc ở Nebraska? Trong các công việc không được coi là đặc biệt nguy hiểm, FLSA quy định độ tuổi tối thiểu bình thường để làm việc trong nông nghiệp là 1... |
| 7 | 5853955 | 0 | 0.6397 | vi | Tuy nhiên, caddie phải từ 14 tuổi trở lên. Người trông trẻ không cần phải có giấy chứng nhận lao động. Tuy nhiên, người trông trẻ phải từ 14 tuổi trở lên. Trẻ vị thành niên 16 v... |
| 8 | 7398107 | 0 | 0.6383 | vi | Một người phải từ 16 tuổi trở lên để làm việc tại The Finish Line. |
| 9 | 5192868 | 0 | 0.6369 | vi | Trả lời Độ tuổi tối thiểu để làm việc khác nhau tùy theo tình huống. Ở Hoa Kỳ, độ tuổi tối thiểu cho hầu hết công việc là 16, mặc dù trong một số trường hợp, nó có thể thấp tới ... |
| 10 | 5829982 | 0 | 0.6365 | vi | Để làm việc cho NATS, bạn phải từ 13 tuổi trở lên vào thời điểm mùa giải bắt đầu. (Chúng tôi nói ngày 1 tháng 7 vì sự đơn giảnÃƒÂ ¢ Ã‚â‚¬Ã‚â „¢ s sake). Mặc dù độ tuổi hợp pháp ... |

### qid `1095874`

- query A (`en`): average salary for firefighters in az
- query B (`vi`): mức lương trung bình cho lính cứu hỏa tính bằng az
- diagnosis: Unclassified; nDCG@10 end=0.0000, mix=100.0000, Δ=100.0000; Recall@10 end=0.0000, mix=100.0000, Δ=100.0000; tokens(a/b)=6/11, len_ratio=0.5455; overlap@10=4; source=evaluate_perquery; focus=best (gain)

Endpoint top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 5940470 | 0 | 0.7228 | vi | Các nhân viên cứu hỏa ở Austin, Texas sẽ kiếm được từ $ 21,641 đến $ 61,210. Mức lương trung bình hàng năm cho một lính cứu hỏa ở Austin là $ 41.424. nó chắc chắn xứng đáng với ... |
| 2 | 1725979 | 0 | 0.7193 | vi | Mức lương trung bình cho một Lính cứu hỏa là C $ 61.437 mỗi năm. |
| 3 | 5529477 | 0 | 0.7183 | vi | Mức lương hàng năm cho lính cứu hỏa trung bình có thể thay đổi dựa trên những yếu tố như kinh nghiệm và vị trí. Tại Hoa Kỳ, mức lương trung bình cho một lính cứu hỏa là $ 42,87 ... |
| 4 | 3035263 | 0 | 0.7168 | vi | Mức lương trung bình của lính cứu hỏa tình nguyện ở Hoa Kỳ là $ 34.059 hoặc mức lương theo giờ tương đương là $ 16. Ngoài ra, họ kiếm được tiền thưởng trung bình là 470 đô la. M... |
| 5 | 2821813 | 0 | 0.7162 | vi | Mức lương trung bình của lính cứu hỏa. Mức lương trung bình cho công việc lính cứu hỏa là $ 39,000. Mức lương trung bình của lính cứu hỏa có thể thay đổi rất nhiều do công ty, v... |
| 6 | 3035265 | 0 | 0.7148 | vi | Mức lương trung bình của lính cứu hỏa tình nguyện ở Hoa Kỳ là $ 34.059 hoặc mức lương theo giờ tương đương là $ 16. Mức lương ước tính dựa trên dữ liệu khảo sát tiền lương được ... |
| 7 | 3910586 | 0 | 0.7143 | vi | Mức lương trung bình hàng năm cho lính cứu hỏa là 48.030 đô la vào tháng 5 năm 2016. Mức lương trung bình là mức lương mà tại đó một nửa số công nhân trong một ngành nghề kiếm đ... |
| 8 | 7556012 | 0 | 0.7141 | vi | Tìm hiểu thêm về Mức lương trung bình của Lính cứu hỏa / Nhân viên y tế ở Arizona trên Đơn giản là Thuê. So sánh mức lương trung bình theo chức danh công việc và bộ kỹ năng. |
| 9 | 1725972 | 0 | 0.7138 | vi | Mức lương trung bình hàng năm cho lính cứu hỏa là 48.030 đô la vào tháng 5 năm 2016. Mức lương trung bình là mức lương mà tại đó một nửa số công nhân trong một ngành nghề kiếm đ... |
| 10 | 4076289 | 0 | 0.7137 | vi | Trung bình Quốc gia. Lính cứu hỏa kiếm được mức lương trung bình hàng năm là 47.270 đô la và mức lương trung bình theo giờ là 22,72 đô la vào tháng 5 năm 2009, theo Cục Thống kê... |

Mixed top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 7556014 | 1 | 0.7229 | vi | Mức lương trung bình cho Sở Cứu hỏa Phoenix - Lính cứu hỏa Arizona: $ 50,193. Sở Cứu hỏa Phoenix - Arizona xu hướng tiền lương dựa trên mức lương được đăng tải ẩn danh bởi các n... |
| 2 | 5940470 | 0 | 0.7107 | vi | Các nhân viên cứu hỏa ở Austin, Texas sẽ kiếm được từ $ 21,641 đến $ 61,210. Mức lương trung bình hàng năm cho một lính cứu hỏa ở Austin là $ 41.424. nó chắc chắn xứng đáng với ... |
| 3 | 7556012 | 0 | 0.7097 | vi | Tìm hiểu thêm về Mức lương trung bình của Lính cứu hỏa / Nhân viên y tế ở Arizona trên Đơn giản là Thuê. So sánh mức lương trung bình theo chức danh công việc và bộ kỹ năng. |
| 4 | 569960 | 0 | 0.6914 | vi | Chức danh Công việc thay thế: Lính cứu hỏa, Lính cứu hỏa. Fire Fighter kiếm được bao nhiêu? Mức lương trung bình hàng năm của Nhân viên cứu hỏa là 44.703 đô la, tính đến ngày 31... |
| 5 | 242629 | 0 | 0.6873 | vi | Mức lương trung bình của lính cứu hỏa. Mức lương trung bình cho công việc lính cứu hỏa ở San Antonio, TX là $ 34,000. Mức lương trung bình của lính cứu hỏa có thể thay đổi rất n... |
| 6 | 6422592 | 0 | 0.6859 | vi | Mức lương trung bình hàng năm cho Lính cứu hỏa là bao nhiêu? Fire Fighter kiếm được bao nhiêu? Mức lương trung bình hàng năm của Nhân viên cứu hỏa là 45.300 đô la, kể từ ngày 02... |
| 7 | 3098840 | 0 | 0.6839 | vi | Trong năm 2011, Cục Thống kê Lao động đã báo cáo mức lương trung bình hàng năm cho lính cứu hỏa ở California là 71.030 đô la. Trong khi đó, mức lương trung bình hàng năm ở Texas... |
| 8 | 6565060 | 0 | 0.6836 | vi | Mức lương trung bình theo giờ của lính cứu hỏa ở Wisconsin. Lính cứu hỏa kiếm được mức lương trung bình hàng giờ là 13,58 đô la. Mức lương hàng giờ thường bắt đầu từ 8,42 đô la ... |
| 9 | 4076289 | 0 | 0.6820 | vi | Trung bình Quốc gia. Lính cứu hỏa kiếm được mức lương trung bình hàng năm là 47.270 đô la và mức lương trung bình theo giờ là 22,72 đô la vào tháng 5 năm 2009, theo Cục Thống kê... |
| 10 | 3035263 | 0 | 0.6818 | vi | Mức lương trung bình của lính cứu hỏa tình nguyện ở Hoa Kỳ là $ 34.059 hoặc mức lương theo giờ tương đương là $ 16. Ngoài ra, họ kiếm được tiền thưởng trung bình là 470 đô la. M... |

### qid `1097223`

- query A (`en`): how many digits account number wells fargo
- query B (`vi`): có bao nhiêu chữ số số tài khoản còn xa
- diagnosis: Unclassified; nDCG@10 end=0.0000, mix=100.0000, Δ=100.0000; Recall@10 end=0.0000, mix=100.0000, Δ=100.0000; tokens(a/b)=7/10, len_ratio=0.7000; overlap@10=0; source=evaluate_perquery; focus=best (gain)

Endpoint top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 5566638 | 0 | 0.7175 | vi | KHÔNG có câu trả lời chính xác cho câu hỏi này. Số lượng chữ số trong tài khoản ngân hàng khác nhau tùy theo ngân hàng và không nhất thiết giống nhau đối với tất cả các tài khoả... |
| 2 | 7975601 | 0 | 0.6964 | vi | Luôn chỉ định số tài khoản ngân hàng của riêng bạn từ 6 đến 17 chữ số trong trường ÃƒÂ ¢ Ã‚â‚¬Ã‚Å “Số tài khoản ngân hàngÃƒÂ ¢ Ã‚â‚¬Ã‚. Tài khoản ngân hàng của Hoa Kỳ có thể có ... |
| 3 | 2696218 | 0 | 0.6956 | vi | Độ dài phổ biến nhất cho số tài khoản ngân hàng là 9, 12 hoặc 10 chữ số. Mặc dù chúng có độ dài từ 4 đến 17 chữ số. Tôi có một cơ sở dữ liệu lớn gồm các số hợp lệ và không có mẫ... |
| 4 | 1333773 | 0 | 0.6943 | vi | Định dạng số tài khoản thích hợp dài 14 chữ số, bắt đầu bằng 746 và bao gồm bao nhiêu số 0 cần thiết để tạo thành 14 chữ số. Số tài khoản của bạn có thể được tìm thấy ở cuối séc... |
| 5 | 1712252 | 0 | 0.6932 | vi | Số tài khoản thường có độ dài từ năm chữ số trở lên với mỗi chữ số đại diện cho một bộ phận của công ty, bộ phận, loại tài khoản, v.v. Chữ số đầu tiên có thể biểu thị loại tài k... |
| 6 | 8140571 | 0 | 0.6887 | vi | Số Tài khoản Ngân hàng HDFC chỉ có 14 (Mười bốn) chữ số. Các số tài khoản trước đây có Mã chi nhánh và Mã sản phẩm trong Số tài khoản. Tuy nhiên, sau đó, định dạng số đã được th... |
| 7 | 5672735 | 0 | 0.6877 | vi | Số thứ chín được sử dụng để xác minh các số trước đó là chính xác. Số tài khoản ngân hàng tiêu chuẩn có độ dài chín hoặc mười chữ số, nhưng có thể dài tới 17 chữ số, điều này ch... |
| 8 | 6670844 | 0 | 0.6872 | vi | Kiểm tra Tài khoản, Tài khoản Thị trường Tiền tệ và Tài khoản Tiết kiệm Y tế (HSA) Định dạng số tài khoản thích hợp dài 14 chữ số, bắt đầu bằng 746 và bao gồm nhiều số 0 nếu cần... |
| 9 | 7057951 | 0 | 0.6866 | vi | Với Visa, MasterCard và Discover, số thẻ tín dụng có 16 chữ số và chữ số thứ 7 đến chữ số 15 là số tài khoản. Điều này để lại chín chữ số cho số tài khoản của bạn ÃƒÂ ¢ Ã‚â‚¬Ã‚â... |
| 10 | 7975599 | 0 | 0.6858 | vi | 12.4.6.5 Số tài khoản cước phí. Độ dài tối đa cho ID trường 6139 (Số tài khoản FXF) là 9; tuy nhiên, cả số tài khoản Cước phí 8 chữ số và 9 chữ số đều được hỗ trợ. Chỉ bốn chữ s... |

Mixed top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 7431837 | 1 | 0.7322 | vi | Có mười chữ số cho tài khoản ngân hàng Wells Fargo. Nếu không chắc chắn về số tài khoản của mình, họ có thể liên hệ với ngân hàng để được hỗ trợ. |
| 2 | 6570149 | 0 | 0.7267 | vi | Số tài khoản Wells Fargo có 10 chữ số. Số tài khoản có thể được tìm thấy trên bảng sao kê tài khoản của bạn hoặc ở phía bên phải của các số định tuyến ở cuối séc của bạn. |
| 3 | 8377236 | 0 | 0.7171 | vi | Số tài khoản Wells Fargo có 10 chữ số. Số tài khoản có thể được tìm thấy trên bảng sao kê tài khoản của bạn hoặc ở phía bên phải của các số định tuyến ở dưới cùng oÃƒÂ ¢ Ã‚â‚¬Ã‚... |
| 4 | 521418 | 0 | 0.6938 | vi | - Số tài khoản của bạn: Số tài khoản đầy đủ của bạn. - Title of Account: Tên tài khoản của bạn xuất hiện trên bảng sao kê. Lưu ý: Có một khoản phí khi nhận chuyển khoản. Để biết... |
| 5 | 8177269 | 0 | 0.6932 | vi | Hướng dẫn 4 bước đơn giản để giải quyết vấn đề Wells Fargo phổ biến này một cách nhanh chóng và hiệu quả bởi GetHuman 1 Xin chào, để đóng tài khoản có số dư bằng 0, hãy làm theo... |
| 6 | 4469480 | 0 | 0.6902 | vi | Cập nhật năm 2017: Việc tìm kiếm số tài khoản của bạn giờ đây dễ dàng hơn bao giờ hết. Từ trang web của Wells Fargo, chỉ cần nhấp vào tài khoản mà bạn muốn để lấy các số. Khi bả... |
| 7 | 52660 | 0 | 0.6881 | vi | Việc tìm kiếm số tài khoản của bạn giờ đây dễ dàng hơn bao giờ hết. Từ trang web của Wells Fargo, chỉ cần nhấp vào tài khoản mà bạn muốn để lấy các số. Khi bản tóm tắt tài khoản... |
| 8 | 4386356 | 0 | 0.6746 | vi | Nếu bạn nhận được một email đáng ngờ yêu cầu số tài khoản của mình, hãy chuyển tiếp email đến reportphish@wellsfargo.com và xóa nó. Nếu bạn cần thêm thông tin, hãy gọi cho chúng... |
| 9 | 53631 | 0 | 0.6728 | vi | 061000227 số định tuyến của WELLS FARGO BANK Kiểm tra số định tuyến của WELLS FARGO BANK để chuyển khoản ngân hàng. 061000227 là số định tuyến của WELLS FARGO BANK. Kiểm tra thô... |
| 10 | 5508191 | 0 | 0.6724 | vi | Cách đóng tài khoản Wells Fargo. Để đóng tài khoản séc không có số dư tại Wells Fargo, bạn có thể gọi cho ngân hàng theo số 800-869-3557 hoặc gửi yêu cầu qua email qua trang web... |

### qid `230725`

- query A (`en`): how far is cedar point from chicago
- query B (`vi`): tuyết tùng cách Chicago bao xa
- diagnosis: Unclassified; nDCG@10 end=0.0000, mix=100.0000, Δ=100.0000; Recall@10 end=0.0000, mix=100.0000, Δ=100.0000; tokens(a/b)=7/6, len_ratio=1.1667; overlap@10=3; source=evaluate_perquery; focus=best (gain)

Endpoint top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 3970237 | 0 | 0.6440 | vi | Có 298 dặm từ St. Louis, MO đến Chicago, IL. Mất khoảng 5 giờ để lái xe đến đó hoặc 3 ngày và 22 giờ để đi bộ đến đó. |
| 2 | 5547280 | 0 | 0.6239 | vi | Câu trả lời của Laary C. Độ tin cậy bình chọn 10,3K. Điều đó phụ thuộc vào phần nào của Nova Scotia mà bạn đang cố gắng tìm khoảng cách. Từ Chicago, IL đến Halifax, Nova Scotia,... |
| 3 | 2324363 | 0 | 0.6206 | vi | Nó là 892 dặm từ Chicago, IL đến Hartford, CT. Bạn sẽ mất khoảng 14 tiếng rưỡi để lái xe. Để biết chỉ đường, lập bản đồ và thời gian, hãy xem Liên kết có liên quan. |
| 4 | 3413114 | 0 | 0.6194 | vi | Khoảng cách từ Minnesota đến Chicago. Khoảng cách từ Minnesota đến Chicago là 780 km. Khoảng cách di chuyển bằng đường hàng không này là 485 dặm. Khoảng cách ngắn nhất giữa Minn... |
| 5 | 5973762 | 0 | 0.6112 | vi | Ghi chú về khoảng cách, tiêu thụ khí và phát thải. Khoảng cách từ Fort Leonard Wood, MO tới Chicago, IL là 442Miles hoặc 711 Km. Bạn có thể đi được khoảng cách này khoảng 6 giờ ... |
| 6 | 2324354 | 0 | 0.6101 | vi | Khoảng cách lái xe giữa Chicago, IL và Hartford, CT là khoảng 895 dặm. Thời gian lái xe sẽ là khoảng 14 giờ 30 phút nếu bạn đi không cần khom lưng trong điều kiện lái xe tốt. |
| 7 | 6540813 | 0 | 0.6093 | vi | tồn tại và là một thay thế của. Khoảng cách giữa Chicago, Illinois, Hoa Kỳ và CancÃƒÆ’Ã‚Âºn, Mexico là 1425 dặm (2293 km). Từ Chicago Illinois đến Cancun Mexico khoảng 1425 dặm ... |
| 8 | 7303435 | 0 | 0.6080 | vi | Tuy nhiên, cứ ba năm một lần hoặc lâu hơn trong mùa đông Chicago lại trải qua một trận bão tuyết lớn hơn có thể tạo ra hơn 10 inch (25 cm) tuyết trong khoảng thời gian từ 1 đến ... |
| 9 | 3234157 | 0 | 0.6060 | vi | Nằm cách hẻm núi Little Cottonwood 6 dặm, Snowbird chỉ cách Sân bay Quốc tế Thành phố Salt Lake 45 phút lái xe tuyệt đẹp, biến việc trượt tuyết và bay trong ngày trở thành hiện ... |
| 10 | 683607 | 0 | 0.6055 | vi | Mùa đông ở Chicago khá thay đổi: Lượng tuyết rơi theo mùa trong thành phố dao động từ 9,8 inch (24,9 cm) (năm 1920ÃƒÂ ¢ Ã‚â‚¬Ã‚â € œ21) lên đến 89,7 inch (228 cm) (năm 1978ÃƒÂ ¢... |

Mixed top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 7737541 | 1 | 0.7156 | vi | Cedar Point cách Chicago 295 dặm. Sẽ mất khoảng 4 giờ 40 phút lái xe. Cedar Point cách Chicago 295 dặm. Sẽ mất khoảng 4 giờ 40 phút lái xe. |
| 2 | 7737547 | 0 | 0.6735 | vi | Khoảng cách lái xe từ Chicago, IL đến Cedar Point, IL. Tổng quãng đường lái xe từ Chicago, IL đến Cedar Point, IL là 103 dặm hoặc 166 km. Chuyến đi của bạn bắt đầu ở Chicago, Il... |
| 3 | 7737549 | 0 | 0.6703 | vi | Ghi chú khoảng cách, tiêu thụ khí và phát thải. Khoảng cách từ Chicago, IL đến Cedar Point, Sandusky, OH là 295Miles hoặc 475 Km. Bạn có thể đi được khoảng cách này khoảng 4 giờ... |
| 4 | 3970237 | 0 | 0.6603 | vi | Có 298 dặm từ St. Louis, MO đến Chicago, IL. Mất khoảng 5 giờ để lái xe đến đó hoặc 3 ngày và 22 giờ để đi bộ đến đó. |
| 5 | 5973762 | 0 | 0.6530 | vi | Ghi chú về khoảng cách, tiêu thụ khí và phát thải. Khoảng cách từ Fort Leonard Wood, MO tới Chicago, IL là 442Miles hoặc 711 Km. Bạn có thể đi được khoảng cách này khoảng 6 giờ ... |
| 6 | 2324363 | 0 | 0.6506 | vi | Nó là 892 dặm từ Chicago, IL đến Hartford, CT. Bạn sẽ mất khoảng 14 tiếng rưỡi để lái xe. Để biết chỉ đường, lập bản đồ và thời gian, hãy xem Liên kết có liên quan. |
| 7 | 6651016 | 0 | 0.6475 | vi | Khoảng cách từ Denver, CO đến Chicago, IL. Tổng khoảng cách từ Denver, CO đến Chicago, IL là 920 dặm. Điều này tương đương với 1ÃƒÂ ¢ Ã‚â‚¬Ã‚â € ° 481 km hoặc 800 hải lý. Chuyến... |
| 8 | 5973759 | 0 | 0.6447 | vi | 1 giờ 18 phút. Khoảng cách từ Fort Leonard Wood, MO tới Chicago, IL là 442Miles hoặc 711 Km. Bạn có thể đi được khoảng cách này khoảng 6 giờ 43 phút. Nếu bạn muốn lập kế hoạch đ... |
| 9 | 5996158 | 0 | 0.6445 | vi | Khoảng cách lái xe giữa Charlotte, NC và Chicago, IL là khoảng 755 dặm. Thời gian lái xe sẽ là khoảng 12 giờ 15 phút nếu bạn đi du lịch không phải st ÃƒÂ ¢ Ã‚â‚¬Ã‚Â¦ trong điều ... |
| 10 | 6651017 | 0 | 0.6398 | vi | Khoảng cách từ Denver, CO đến Chicago, IL. Tổng khoảng cách từ Denver, CO đến Chicago, IL là 920 dặm. Điều này tương đương với 1ÃƒÂ ¢ Ã‚â‚¬Ã‚â € ° 481 km hoặc 800 hải lý. Chuyến... |

### qid `249821`

- query A (`en`): how long does a mri of the knee take
- query B (`vi`): đầu gối mất bao lâu
- diagnosis: TranslationDivergence; nDCG@10 end=0.0000, mix=100.0000, Δ=100.0000; Recall@10 end=0.0000, mix=100.0000, Δ=100.0000; tokens(a/b)=9/5, len_ratio=1.8000; overlap@10=2; source=evaluate_perquery; focus=best (gain)

Endpoint top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 6438625 | 0 | 0.7341 | vi | Số phiếu tự tin 261K. Phẫu thuật thay thế đầu gối bao gồm việc thay thế một phần đầu gối bị mất khối lượng xương. Có một số phương pháp thay thế đầu gối khác nhau và chúng thườn... |
| 2 | 6046542 | 0 | 0.7254 | vi | 2 bác sĩ thống nhất: Còn tùy: Phục hồi hoàn toàn sau khi thay khớp gối thường mất hơn một năm. Hầu hết bệnh nhân giảm việc sử dụng thuốc giảm đau trong vòng hai tuần sau khi tha... |
| 3 | 7351122 | 0 | 0.7215 | vi | Thời gian để phục hồi. Thông thường, phải mất 2 đến 4 tuần để đầu gối bị giãn hoàn toàn. Vật lý trị liệu và nghỉ ngơi là hai phương pháp chính để điều trị tình trạng này. Thuốc ... |
| 4 | 4627496 | 0 | 0.7182 | vi | Mốc thời gian để phục hồi Đầu gối của bạn sẽ khá hơn, nhưng có thể mất nhiều thời gian hơn so với chấn thương thể thao trung bình hoặc gãy xương. Quá trình hồi phục hoàn toàn có... |
| 5 | 3533493 | 0 | 0.7180 | vi | Tất nhiên, mỗi người là khác nhau và thời gian phục hồi có thể khác nhau, tùy thuộc vào một số yếu tố. Thời gian hồi phục hoàn toàn điển hình sau khi thay toàn bộ đầu gối là từ ... |
| 6 | 5233099 | 0 | 0.7153 | vi | Đừng đợi đến gặp bác sĩ phẫu thuật nếu bất kỳ lúc nào bạn cảm thấy đau, sưng, cứng hoặc cử động bất thường ở đầu gối. Tất nhiên, mỗi người là khác nhau và thời gian phục hồi có ... |
| 7 | 7795883 | 0 | 0.7147 | vi | Tất nhiên, mỗi người là khác nhau và thời gian phục hồi có thể khác nhau, tùy thuộc vào một số yếu tố. Thời gian hồi phục hoàn toàn điển hình sau khi thay toàn bộ đầu gối là từ ... |
| 8 | 5681648 | 0 | 0.7143 | vi | Câu hỏi liên quan đến sức khỏe trong chủ đề Tình trạng Bệnh tật. Chúng tôi đã tìm thấy một số câu trả lời như dưới đây cho câu hỏi này Mất bao lâu để chữa lành đầu gối bị gãy, b... |
| 9 | 1206145 | 0 | 0.7141 | vi | Tất nhiên, mỗi người là khác nhau và thời gian phục hồi có thể khác nhau tùy thuộc vào một số yếu tố. Một sự hồi phục hoàn toàn điển hình sau khi thay toàn bộ đầu gối là từ 3 đế... |
| 10 | 1990378 | 0 | 0.7139 | vi | Thời gian phục hồi ngắn hạn trung bình cho một lần thay toàn bộ đầu gối là khoảng 12 tuần. Phục hồi lâu dài liên quan đến việc chữa lành hoàn toàn vết thương phẫu thuật và các m... |

Mixed top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 6006595 | 1 | 0.7439 | vi | Đánh giá Mới nhất Cũ nhất. Câu trả lời hay nhất: Chụp MRI đầu gối sẽ mất khoảng 30-60 phút, tùy thuộc vào máy được sử dụng và số lượng chuỗi hình ảnh được thực hiện. Bạn sẽ cần ... |
| 2 | 5484229 | 0 | 0.7048 | vi | Ít nhất 6 tuần nhưng có thể lâu nhất là 12 tuần. Bạn cần phải làm từ từ cho đến khi nó hoàn toàn lành lại nếu không bạn có thể làm hỏng nó một lần nữa. Chúng dễ dàng bị hư hỏng ... |
| 3 | 7523834 | 0 | 0.7043 | vi | Chụp MRI đầu gối sử dụng nam châm và sóng vô tuyến để ghi lại hình ảnh bên trong cơ thể bạn. Không giống như chụp X-quang và chụp CT, MRI không sử dụng bức xạ. Quy trình chụp MR... |
| 4 | 2153743 | 0 | 0.6998 | vi | Một nghiên cứu thứ hai, nhỏ hơn đã kiểm tra những bệnh nhân từng bị đau đầu gối nhẹ trong trung bình 14 tháng. 21 Mỗi khớp gối được chụp MRI để đánh giá tổn thương khớp và sau đ... |
| 5 | 1990378 | 0 | 0.6909 | vi | Thời gian phục hồi ngắn hạn trung bình cho một lần thay toàn bộ đầu gối là khoảng 12 tuần. Phục hồi lâu dài liên quan đến việc chữa lành hoàn toàn vết thương phẫu thuật và các m... |
| 6 | 1252959 | 0 | 0.6905 | vi | Thời gian phục hồi ngắn hạn trung bình cho một lần thay toàn bộ đầu gối là 6 đến 12 tuần. Phục hồi lâu dài liên quan đến việc chữa lành hoàn toàn vết thương phẫu thuật và các mô... |
| 7 | 1626822 | 0 | 0.6897 | vi | Tổn thương đầu gối trên MRI Có khả năng trầm trọng hơn dẫn đến viêm khớp. Tổn thương đầu gối rõ ràng trên hình ảnh chụp cộng hưởng từ (MRI) và trở nên tồi tệ hơn trong 12 đến 48... |
| 8 | 4627492 | 0 | 0.6897 | vi | Bệnh nhân: Tôi bị đập vào đầu gối trong một trận bóng đá ở trường trung học vào tháng 10 năm 2010. Nó bị thôi miên và làm trật khớp xương bánh chè của tôi. Tôi phải phẫu thuật n... |
| 9 | 6438625 | 0 | 0.6885 | vi | Số phiếu tự tin 261K. Phẫu thuật thay thế đầu gối bao gồm việc thay thế một phần đầu gối bị mất khối lượng xương. Có một số phương pháp thay thế đầu gối khác nhau và chúng thườn... |
| 10 | 6197992 | 0 | 0.6884 | vi | Thời gian phục hồi ngắn hạn trung bình cho một lần thay toàn bộ đầu gối là khoảng 12 tuần. Phục hồi lâu dài Phục hồi lâu dài liên quan đến việc chữa lành hoàn toàn vết thương ph... |

### qid `277701`

- query A (`en`): how many calories in publix chicken tender subway
- query B (`vi`): bao nhiêu calo trong tàu điện ngầm gà mềm publix
- diagnosis: Unclassified; nDCG@10 end=0.0000, mix=100.0000, Δ=100.0000; Recall@10 end=0.0000, mix=100.0000, Δ=100.0000; tokens(a/b)=8/11, len_ratio=0.7273; overlap@10=4; source=evaluate_perquery; focus=best (gain)

Endpoint top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 7476516 | 0 | 0.6513 | vi | Thông tin về lượng calo cụ thể theo thành phần từ công thức của chúng tôi: Lượng calo trong 6 tàu điện ngầm Black Forest Ham Sub (không phô mai) Calo: 529, Chất béo: 7g, Carb: 9... |
| 2 | 6153908 | 0 | 0.6503 | vi | Tùy chọn Khỏe mạnh nhất trên Menu Tàu điện ngầm. 1 con gà nướng trong lò (trên bánh mì Ý): 330 calo, 4,5 gam chất béo, 22 gam protein. 2 Gà kiểu Rotisserie: 350 calo, 6 gam chất... |
| 3 | 7117600 | 0 | 0.6462 | vi | Bao nhiêu calo trong Tàu điện ngầm 6 gram chất béo hoặc ít hơn: Gỏi đôi gà băm, không tẩm gia vị |
| 4 | 7103020 | 0 | 0.6461 | vi | Quầy Calo của Tàu điện ngầm. Đồ ăn. 6 Giăm bông (Thịt nguội thái lát và rau tiêu chuẩn) 6 Ức gà nướng trong lò (Ức gà nướng không xương và các loại rau tiêu chuẩn) 6 Thịt bò nướ... |
| 5 | 109669 | 0 | 0.6452 | vi | Tàu điện ngầm: Cược Kém Món gà 6 inch và thịt xông khói Melt là một lựa chọn béo bở tại một nhà hàng nổi tiếng với các lựa chọn tốt cho sức khỏe. Chiếc phụ 6 inch này nặng 600 c... |
| 6 | 3549746 | 0 | 0.6413 | vi | Nếu bạn đang ăn kiêng, hoặc chỉ đơn giản là theo dõi lượng calo nạp vào cơ thể, bạn nên tránh những nguy cơ về calo trên tàu điện ngầm sau đây. Cá nhân tôi không thể tin rằng có... |
| 7 | 7855993 | 0 | 0.6383 | vi | Bao nhiêu calo trong một tàu điện ngầm 6 'thịt viên marinara? Trả lời 480. Máy đếm calo trong tàu điện ngầm (calorielab) dinh dưỡng, carbohydrate và calo trong tàu điện ngầm cà ... |
| 8 | 8044213 | 0 | 0.6370 | vi | Thông tin dinh dưỡng, thông tin về chế độ ăn uống và lượng calo trong gà trâu trên bánh mì dẹt từ tàu điện ngầm |
| 9 | 109670 | 0 | 0.6345 | vi | Tàu điện ngầm: Cá cược tốt hơn. Giữ lượng calo cá nhân của bạn ở mức thấp với Bánh Sandwich Black Forest Ham 6 inch. Chiếc phụ 6 inch này có 290 calo, 4,5 g chất béo, 1g chất bé... |
| 10 | 5819229 | 0 | 0.6291 | vi | Có 430 calo trong 1 khẩu phần bánh sandwich của Tàu điện ngầm 6 Gà tây thịt xông khói Bơ. Phân hủy calo: 39% chất béo, 40% carbs, 22% protein. |

Mixed top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 7360038 | 1 | 0.6635 | vi | Có 620 calo trong 1 khẩu phần sandwich Publix 12 Chicken Tender Sub Sandwich. Phân hủy calo: 20% chất béo, 57% carbs, 23% protein. |
| 2 | 7103020 | 0 | 0.6433 | vi | Quầy Calo của Tàu điện ngầm. Đồ ăn. 6 Giăm bông (Thịt nguội thái lát và rau tiêu chuẩn) 6 Ức gà nướng trong lò (Ức gà nướng không xương và các loại rau tiêu chuẩn) 6 Thịt bò nướ... |
| 3 | 5838181 | 0 | 0.6382 | vi | Có 510 calo trong 1 khẩu phần phụ của Subway 6 Chicken Parmesan. Phân hủy calo: 30% chất béo, 49% carbs, 21% protein. |
| 4 | 5819229 | 0 | 0.6348 | vi | Có 430 calo trong 1 khẩu phần bánh sandwich của Tàu điện ngầm 6 Gà tây thịt xông khói Bơ. Phân hủy calo: 39% chất béo, 40% carbs, 22% protein. |
| 5 | 7408554 | 0 | 0.6335 | vi | Có 330 calo trong 1 khẩu phần nướng của Xúc xích Ý Thịt heo Publix Mild. Phân hủy calo: 62% chất béo, 3% carbs, 35% protein. Xúc xích liên quan từ Publix: |
| 6 | 5545858 | 0 | 0.6263 | vi | Mặc dù không được chỉ định trong tệp thông tin dinh dưỡng của Subway, phiên bản footlongÃƒÂ ¢ Ã‚â € žÃ‚Â ¢ của bất kỳ loại bánh mì thông thường nào cung cấp lượng calo và chất b... |
| 7 | 7476516 | 0 | 0.6250 | vi | Thông tin về lượng calo cụ thể theo thành phần từ công thức của chúng tôi: Lượng calo trong 6 tàu điện ngầm Black Forest Ham Sub (không phô mai) Calo: 529, Chất béo: 7g, Carb: 9... |
| 8 | 98751 | 0 | 0.6238 | vi | Có 450 calo trong một khẩu phần 1 bánh sandwich của Subway 6 Chipotle Chicken & Cheese. Phân hủy calo: 36% chất béo, 39% carbs, 24% protein. Bánh mì liên quan từ tàu điện ngầm: |
| 9 | 109669 | 0 | 0.6182 | vi | Tàu điện ngầm: Cược Kém Món gà 6 inch và thịt xông khói Melt là một lựa chọn béo bở tại một nhà hàng nổi tiếng với các lựa chọn tốt cho sức khỏe. Chiếc phụ 6 inch này nặng 600 c... |
| 10 | 3549740 | 0 | 0.6172 | vi | Cá nhân tôi không thể tin rằng có thể có nhiều calo như vậy trong một số bánh sandwich tàu điện ngầm: thịt đôi người Ý BMT ÃƒÂ ¢ Ã‚â‚¬Ã‚â € œ 6 inch: 630 calo, chân dài: 1260 ca... |

### qid `288702`

- query A (`en`): how many miles between mobile and destin beach
- query B (`vi`): bao nhiêu dặm giữa bãi biển di động và bãi biển destin
- diagnosis: Unclassified; nDCG@10 end=0.0000, mix=100.0000, Δ=100.0000; Recall@10 end=0.0000, mix=100.0000, Δ=100.0000; tokens(a/b)=8/12, len_ratio=0.6667; overlap@10=7; source=evaluate_perquery; focus=best (gain)

Endpoint top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 1143809 | 0 | 0.6036 | vi | Bao nhiêu Miles giữa bãi biển daytona và bãi biển delray Florida? Khoảng cách giữa Bãi biển Daytona, FL và Bãi biển Delray, Fl là khoảng 210 dặm. Hãy nhớ rằng điều này phụ thuộc... |
| 2 | 2989529 | 0 | 0.5947 | vi | Khoảng cách giữa Destin và Bãi biển thành phố Panama theo đường thẳng là 38 dặm hoặc 61,14 Kilomét. |
| 3 | 4214338 | 0 | 0.5889 | vi | Bến du thuyền là một bãi biển đô thị tự nhiên dọc theo bờ biển Coramandel trên Vịnh Bengal. Chủ yếu là cát, bãi biển kéo dài khoảng 13 km (8,1 mi), chạy từ gần Pháo đài St. Geor... |
| 4 | 7484618 | 0 | 0.5791 | vi | Ghi chú về khoảng cách, tiêu thụ khí và phát thải. Khoảng cách từ Bãi biển Miramar, FL đến Destin, FL là 9Miles hoặc 14 Km. Bạn có thể nhận được khoảng cách này khoảng 17 phút. ... |
| 5 | 6402725 | 0 | 0.5746 | vi | Ghi chú về khoảng cách, tiêu thụ khí và phát thải. Khoảng cách từ Davenport, FL tới Daytona Beach, FL là 91Miles hoặc 146 Km. Bạn có thể đi được khoảng cách này khoảng 1 giờ 34 ... |
| 6 | 2919041 | 0 | 0.5738 | vi | Khoảng cách giữa Bãi biển Daytona và Bãi biển St Augustine theo đường thẳng là 50 dặm hoặc 80,45 Kilomét. 1 Chỉ đường Lái xe & Thời gian Lái xe từ Bãi biển Daytona đến Bãi biển ... |
| 7 | 69688 | 0 | 0.5738 | vi | Vùng lân cận Destin. Với diện tích khoảng tám dặm vuông, Destin thực sự không phải là một thành phố khó điều động. Có 12 điểm truy cập công cộng cho những du khách muốn dành một... |
| 8 | 3038344 | 0 | 0.5729 | vi | Khoảng cách giữa Destin FL và Pensacola FL. Khoảng cách từ Destin đến Pensacola là 76 km đường bộ. Đường mất khoảng 1 giờ 4 phút và đi qua Bãi biển Fort Walton, Mary Esther và V... |
| 9 | 2940972 | 0 | 0.5679 | vi | Dặm Từ Portland đến Ã‚â‚¬Ã‚Â¦. Tới Seaside ÃƒÂ ¢ Ã‚â‚¬Ã‚â € œ 79 dặm Tới Cannon Beach ÃƒÂ ¢ Ã‚â‚¬Ã‚â € œ 79 dặm Tới Manzanita ÃƒÂ ¢ Ã‚â‚¬Ã‚â € œ 110 dặm Tới Tillamook (qua Hwy 6... |
| 10 | 6402718 | 0 | 0.5668 | vi | 40 phút. Khoảng cách từ Davenport, FL tới Daytona Beach, FL là 91Miles hoặc 146 Km. Bạn có thể đi được khoảng cách này khoảng 1 giờ 34 phút. Nếu bạn muốn lập kế hoạch đi du lịch... |

Mixed top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 7484614 | 1 | 0.6014 | vi | Khoảng cách lái xe từ Mobile, AL đến Destin, FL. Tổng quãng đường lái xe từ Mobile, AL đến Destin, FL là 108 dặm hoặc 174 km. Chuyến đi của bạn bắt đầu ở Mobile, Alabama. Nó kết... |
| 2 | 7484621 | 0 | 0.6006 | vi | Bản đồ đường đi từ Mobile, AL đến Destin, FL. Bản đồ tuyến đường tối ưu giữa Mobile, AL và Destin, FL. Lộ trình này sẽ là khoảng 108 Dặm. Thông tin về tuyến đường lái xe (khoảng... |
| 3 | 2989529 | 0 | 0.5923 | vi | Khoảng cách giữa Destin và Bãi biển thành phố Panama theo đường thẳng là 38 dặm hoặc 61,14 Kilomét. |
| 4 | 7484618 | 0 | 0.5909 | vi | Ghi chú về khoảng cách, tiêu thụ khí và phát thải. Khoảng cách từ Bãi biển Miramar, FL đến Destin, FL là 9Miles hoặc 14 Km. Bạn có thể nhận được khoảng cách này khoảng 17 phút. ... |
| 5 | 1143809 | 0 | 0.5807 | vi | Bao nhiêu Miles giữa bãi biển daytona và bãi biển delray Florida? Khoảng cách giữa Bãi biển Daytona, FL và Bãi biển Delray, Fl là khoảng 210 dặm. Hãy nhớ rằng điều này phụ thuộc... |
| 6 | 6402725 | 0 | 0.5796 | vi | Ghi chú về khoảng cách, tiêu thụ khí và phát thải. Khoảng cách từ Davenport, FL tới Daytona Beach, FL là 91Miles hoặc 146 Km. Bạn có thể đi được khoảng cách này khoảng 1 giờ 34 ... |
| 7 | 3038344 | 0 | 0.5752 | vi | Khoảng cách giữa Destin FL và Pensacola FL. Khoảng cách từ Destin đến Pensacola là 76 km đường bộ. Đường mất khoảng 1 giờ 4 phút và đi qua Bãi biển Fort Walton, Mary Esther và V... |
| 8 | 6402718 | 0 | 0.5748 | vi | 40 phút. Khoảng cách từ Davenport, FL tới Daytona Beach, FL là 91Miles hoặc 146 Km. Bạn có thể đi được khoảng cách này khoảng 1 giờ 34 phút. Nếu bạn muốn lập kế hoạch đi du lịch... |
| 9 | 3389167 | 0 | 0.5695 | vi | Given là khoảng cách được tính toán cũng như thời gian lái xe, không tính đến điều kiện lái xe, giao thông, v.v. Khoảng cách giữa Miami, Florida và ÃƒÂ ¢ Ã‚â‚¬Ã‚Â¦ Daytona Beach... |
| 10 | 2940972 | 0 | 0.5655 | vi | Dặm Từ Portland đến Ã‚â‚¬Ã‚Â¦. Tới Seaside ÃƒÂ ¢ Ã‚â‚¬Ã‚â € œ 79 dặm Tới Cannon Beach ÃƒÂ ¢ Ã‚â‚¬Ã‚â € œ 79 dặm Tới Manzanita ÃƒÂ ¢ Ã‚â‚¬Ã‚â € œ 110 dặm Tới Tillamook (qua Hwy 6... |

### qid `299094`

- query A (`en`): how many units for frown lines
- query B (`vi`): có bao nhiêu đơn vị cho đường cau mày
- diagnosis: Unclassified; nDCG@10 end=0.0000, mix=100.0000, Δ=100.0000; Recall@10 end=0.0000, mix=100.0000, Δ=100.0000; tokens(a/b)=6/9, len_ratio=0.6667; overlap@10=4; source=evaluate_perquery; focus=best (gain)

Endpoint top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 7377686 | 0 | 0.5680 | vi | Tóm tắt: Đường nâng chân mày (2-5 chiếc), Đường trán (10-30 chiếc), Đường nhăn hoặc Glabellar (10-25 chiếc), Đôi chân lông mày (5-15 chiếc mỗi bên cạnh), Bunny hoặc Nasals Lines... |
| 2 | 6100792 | 0 | 0.5571 | vi | 21 đơn vị Dysport là không nhiều đối với một Glabella (khu vực cau mày) nhưng 21 đơn vị Botox là tốt trong phạm vi bình thường. Bạn nên gọi cho văn phòng nơi bạn đã được điều tr... |
| 3 | 6530140 | 0 | 0.5524 | vi | 20 đơn vị cho 11 đường giữa lông mày. Nếu không có hoạt động cơ ở khu vực đó khi bạn cố gắng cau có, thì Botox đã thực hiện những gì nó được yêu cầu, nhưng hãy đợi một vài ngày ... |
| 4 | 5495414 | 0 | 0.5423 | vi | Thông thường, đối với đường dọc môi trên cần 4-8 đơn vị, đối với nụ cười hở lợi 2 bên 2-4 đơn vị, đường Marionette 2-8 đơn vị và má lúm đồng tiền ở cằm 4-8 đơn vị. Bây giờ để tí... |
| 5 | 6255439 | 0 | 0.5375 | vi | Thông thường, các đường dọc giữa lông mày được tiêm 13-30 đơn vị Botox trong khi các đường ngang trán cần 10-20 đơn vị, các chân mày cần 8-20 đơn vị mỗi bên. và các dòng thỏ cần... |
| 6 | 1822995 | 0 | 0.5367 | vi | Một cái cau mày mà chỉ có khóe miệng và môi dưới đi xuống sử dụng ba cặp cơ. Một cặp hạ môi dưới, và hai cặp hạ khóe môi. Số cặp cơ tối thiểu cho một nụ cười là năm và cho một c... |
| 7 | 5203131 | 0 | 0.5276 | vi | 1 Ở mi mắt, là khu vực giữa hai mắt thường ở phụ nữ chiếm từ 16-28 đơn vị và ở nam giới có từ 18-30 đơn vị. 2 Chỉnh sửa các vết chân chim hai bên 18-24 đơn vị, 3 Đường ngang trá... |
| 8 | 5866116 | 0 | 0.5246 | vi | Cô giáo 38 y / o với đường trán 20 đơn vị Botox đã thư giãn cơ trán, làm mềm các đường nét. Giáo viên 38 tuổi với những đường nét cười đáng chú ý quanh mắt 10 đơn vị Botox mỗi b... |
| 9 | 5670455 | 0 | 0.5232 | vi | Sau đây là phạm vi của Botox trên mỗi đơn vị chúng tôi sử dụng trên mỗi khu vực. Ở mi mắt, là vùng giữa hai mắt thường ở phụ nữ mất từ ​​16-28 đơn vị và ở nam giới mất từ ​​18-3... |
| 10 | 2969995 | 0 | 0.5212 | vi | Để đưa ra một số quan điểm, một người bình thường cần khoảng 20 đơn vị để điều trị nếp nhăn ngang trán, 2 đến 10 đơn vị để điều trị vết chân chim quanh mắt và 25 đơn vị để điều ... |

Mixed top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 7377683 | 1 | 0.6007 | vi | Tùy thuộc vào kích thước và sức mạnh của các cơ này mà bác sĩ của chúng tôi sẽ xác định số lượng đơn vị là cần thiết. Lượng Botox trung bình được sử dụng cho các đường nhăn Frow... |
| 2 | 259608 | 0 | 0.5604 | vi | UNF (Unified Fine) cũng giống như UNC ngoại trừ việc đối với mỗi kích thước của bu lông, số lượng ren trên mỗi inch cao hơn, do đó, ren sẽ mịn hơn. UNEF (Unified Extra Fine) cũn... |
| 3 | 7377686 | 0 | 0.5602 | vi | Tóm tắt: Đường nâng chân mày (2-5 chiếc), Đường trán (10-30 chiếc), Đường nhăn hoặc Glabellar (10-25 chiếc), Đôi chân lông mày (5-15 chiếc mỗi bên cạnh), Bunny hoặc Nasals Lines... |
| 4 | 6255439 | 0 | 0.5434 | vi | Thông thường, các đường dọc giữa lông mày được tiêm 13-30 đơn vị Botox trong khi các đường ngang trán cần 10-20 đơn vị, các chân mày cần 8-20 đơn vị mỗi bên. và các dòng thỏ cần... |
| 5 | 8545760 | 0 | 0.5430 | vi | 1 20 đơn vị lõi. 2 5 đơn vị tự chọn, bao gồm: 3 ít nhất 3 đơn vị từ các đơn vị tự chọn được liệt kê dưới đây. tối đa 2 đơn vị từ bất kỳ Gói Đào tạo được chứng thực hoặc khóa học... |
| 6 | 5203131 | 0 | 0.5421 | vi | 1 Ở mi mắt, là khu vực giữa hai mắt thường ở phụ nữ chiếm từ 16-28 đơn vị và ở nam giới có từ 18-30 đơn vị. 2 Chỉnh sửa các vết chân chim hai bên 18-24 đơn vị, 3 Đường ngang trá... |
| 7 | 7377682 | 0 | 0.5416 | vi | Số lượng đơn vị (hoặc ÃƒÂ ¢ Ã‚â‚¬Ã‚Å “ccÃƒÂ ¢ Ã‚â‚¬Ã‚) bạn cần sẽ phụ thuộc vào độ sâu của đường nhăn / nếp nhăn của bạn, tại thời điểm đó chuyên gia của chúng tôi sẽ đề xuất mộ... |
| 8 | 36102 | 0 | 0.5402 | vi | TransUnion cung cấp 21 điểm tín dụng FICO khác nhau cho người cho vay, sáu phiên bản của điểm chung và tối đa bốn thế hệ của điểm ngành. Một số điểm FICO cũ hơn chỉ dành cho nhữ... |
| 9 | 5495414 | 0 | 0.5397 | vi | Thông thường, đối với đường dọc môi trên cần 4-8 đơn vị, đối với nụ cười hở lợi 2 bên 2-4 đơn vị, đường Marionette 2-8 đơn vị và má lúm đồng tiền ở cằm 4-8 đơn vị. Bây giờ để tí... |
| 10 | 7377685 | 0 | 0.5394 | vi | DÒNG FROWN (11). Botox điều trị cho các đường Frown Lines (11). Còn được gọi là dòng glabella dùng để tiêm Botox đã nhận được sự chấp thuận của FDA. Nó phát triển theo thời gian... |

### qid `412319`

- query A (`en`): is hocking college in ohio a technical school
- query B (`vi`): đang theo học đại học ở ohio một trường kỹ thuật
- diagnosis: Unclassified; nDCG@10 end=0.0000, mix=100.0000, Δ=100.0000; Recall@10 end=0.0000, mix=100.0000, Δ=100.0000; tokens(a/b)=8/11, len_ratio=0.7273; overlap@10=5; source=evaluate_perquery; focus=best (gain)

Endpoint top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 3880332 | 0 | 0.6762 | vi | Chào mừng bạn đến với trang chủ trực tuyến để được tư vấn học tập bậc đại học trong Trường Cao đẳng Kỹ thuật tại Đại học Bang Ohio! Trang web này dành riêng để cung cấp cho bạn ... |
| 2 | 2780310 | 0 | 0.6630 | vi | Công nghệ hàn và chế tạo OTC được trang bị thiết bị hàn mới nhất. Các lớp học chuyên nghiệp của chúng tôi sẽ chuẩn bị cho bạn một sự nghiệp bổ ích. Trường Cao đẳng Kỹ thuật Ohio... |
| 3 | 7516111 | 0 | 0.6562 | vi | Kỹ thuật hàn. Chương trình Kỹ thuật hàn, một phần của Khoa Khoa học và Kỹ thuật Vật liệu, được thiết kế để đào tạo các kỹ sư hàn đáp ứng những thách thức sản xuất của thế kỷ 21.... |
| 4 | 7511886 | 0 | 0.6538 | vi | Chương trình Kỹ thuật. http://www.cofo.edu/Page/Academics/Academic-Programs/Engineering.1608.html. Kỹ thuật tại College of the Ozarks dự kiến ​​bắt đầu vào mùa thu năm 2016. Trư... |
| 5 | 8628276 | 0 | 0.6535 | vi | Mỗi năm, trung bình có 399 sinh viên tốt nghiệp từ các trường kỹ thuật phẫu thuật ở Ohio. Ohio có 25 trường đào tạo kỹ thuật viên phẫu thuật để bạn lựa chọn nếu bạn muốn theo họ... |
| 6 | 1412261 | 0 | 0.6524 | vi | Chương trình Kỹ thuật hàn, một phần của Khoa Khoa học và Kỹ thuật Vật liệu, được thiết kế để đào tạo các kỹ sư hàn đáp ứng những thách thức sản xuất của thế kỷ 21. Chương trình ... |
| 7 | 703691 | 0 | 0.6423 | vi | Chao buổi chiêu! Tôi là sinh viên Kỹ thuật Công nghiệp tại Đại học Wayne State ở Detroit! Tôi hoàn toàn thích nó ở Detroit và triển vọng việc làm thật tuyệt vời (đặc biệt nếu b... |
| 8 | 6390646 | 0 | 0.6355 | vi | Tôi đang đi học để trở thành một kỹ thuật viên thú y, nhưng tôi biết một cô gái trong lớp của tôi, người đã là kỹ thuật viên thú y ở kentrucky, nhưng vẫn được chứng nhận để làm ... |
| 9 | 3299435 | 0 | 0.6344 | vi | Kỹ thuật Cơ khí và Hàng không vũ trụ. Bang Ohio cam kết giúp sinh viên hoàn thành chương trình học của mình. Năm 2013, tỷ lệ tốt nghiệp sáu năm của Bang Ohio là 83,2%, tăng từ 7... |
| 10 | 3880326 | 0 | 0.6327 | vi | Tân sinh viên năm nhất. Văn phòng tuyển sinh của Ohio StateÃƒÂ ¢ Ã‚â‚¬Ã‚â là nguồn lực chính cho quá trình đăng ký và xử lý việc ghi danh trực tiếp cho sinh viên đại học vào trư... |

Mixed top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 7681853 | 1 | 0.6871 | vi | Trường Cao đẳng Kỹ thuật Hocking. Trường Y tế & Điều dưỡng. Trường Cao đẳng Kỹ thuật Hocking là một học viện Công lập có cơ sở tại Nelsonville, Ohio. Cơ quan được Ủy ban Điều dư... |
| 2 | 2780310 | 0 | 0.6800 | vi | Công nghệ hàn và chế tạo OTC được trang bị thiết bị hàn mới nhất. Các lớp học chuyên nghiệp của chúng tôi sẽ chuẩn bị cho bạn một sự nghiệp bổ ích. Trường Cao đẳng Kỹ thuật Ohio... |
| 3 | 7681849 | 0 | 0.6696 | vi | Hocking College là một lựa chọn hợp lý ở Đông Nam Ohio, nơi sinh viên truyền thống, sinh viên phi truyền thống và cựu chiến binh có được các kỹ năng cần thiết để bắt đầu sự nghi... |
| 4 | 7516111 | 0 | 0.6457 | vi | Kỹ thuật hàn. Chương trình Kỹ thuật hàn, một phần của Khoa Khoa học và Kỹ thuật Vật liệu, được thiết kế để đào tạo các kỹ sư hàn đáp ứng những thách thức sản xuất của thế kỷ 21.... |
| 5 | 7681847 | 0 | 0.6429 | vi | Nhấp vào bản đồ bên dưới để xem địa chỉ khuôn viên trường thực tế và nhận chỉ đường lái xe đến: Trường Cao đẳng Kỹ thuật Hocking, 3301 Hocking Parkway, Nelsonville, Ohio 45764. ... |
| 6 | 1412261 | 0 | 0.6403 | vi | Chương trình Kỹ thuật hàn, một phần của Khoa Khoa học và Kỹ thuật Vật liệu, được thiết kế để đào tạo các kỹ sư hàn đáp ứng những thách thức sản xuất của thế kỷ 21. Chương trình ... |
| 7 | 7511886 | 0 | 0.6352 | vi | Chương trình Kỹ thuật. http://www.cofo.edu/Page/Academics/Academic-Programs/Engineering.1608.html. Kỹ thuật tại College of the Ozarks dự kiến ​​bắt đầu vào mùa thu năm 2016. Trư... |
| 8 | 7681846 | 0 | 0.6319 | vi | Được thành lập vào những năm 1960 tại Nelsonville, OH, Hocking College đã phát triển từ một trường dạy nghề thành một trường cao đẳng 2 năm lớn. Trường cung cấp các chương trình... |
| 9 | 3880332 | 0 | 0.6306 | vi | Chào mừng bạn đến với trang chủ trực tuyến để được tư vấn học tập bậc đại học trong Trường Cao đẳng Kỹ thuật tại Đại học Bang Ohio! Trang web này dành riêng để cung cấp cho bạn ... |
| 10 | 1333020 | 0 | 0.6160 | vi | Mặc dù hầu hết mọi người có thể nghĩ về hàn theo quy trình, nhưng nó thực sự là một ngành kỹ thuật phức tạp liên quan đến các khía cạnh của khoa học vật liệu, thiết kế, kiểm tra... |

### qid `424408`

- query A (`en`): is spirit airlines grounded in houston, tx
- query B (`vi`): là hãng hàng không linh có căn cứ ở houston, tx
- diagnosis: Unclassified; nDCG@10 end=0.0000, mix=100.0000, Δ=100.0000; Recall@10 end=0.0000, mix=100.0000, Δ=100.0000; tokens(a/b)=7/11, len_ratio=0.6364; overlap@10=2; source=evaluate_perquery; focus=best (gain)

Endpoint top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 820648 | 0 | 0.6491 | vi | Vận tải, Hàng không, Vận tải Hàng không, Theo lịch trình. Giới thiệu: American Airlines, Inc., hoạt động với tên gọi American Airlines (AA), là một hãng hàng không của Hoa Kỳ và... |
| 2 | 3106653 | 0 | 0.6373 | vi | Hãng hàng không Tây Nam. Đối với các mục đích sử dụng khác, xem Southwest Airlines (định hướng). Southwest Airlines Co. (NYSE: LUV) là một hãng hàng không lớn của Hoa Kỳ, hãng h... |
| 3 | 8576325 | 0 | 0.6365 | vi | tồn tại và là một thay thế của. Southwest Airlines hiện là hãng hàng không lớn nhất bay từ Sân bay Hobby ở Houston, Texas. Delta Air Lines, American Airlines, Air Tran Airlines ... |
| 4 | 7205452 | 0 | 0.6357 | vi | Vận tải, Hàng không, Vận tải Hàng không, Theo lịch trình. American Airlines, Inc., hoạt động với tên gọi American Airlines (AA), là một hãng hàng không của Hoa Kỳ và là công ty ... |
| 5 | 1431392 | 0 | 0.6354 | vi | Southwest Airlines Co. (NYSE: LUV) là một hãng hàng không lớn của Hoa Kỳ và là hãng hàng không giá rẻ lớn nhất thế giới, có trụ sở chính tại Dallas, Texas. vào tháng 12 năm 2014... |
| 6 | 3957520 | 0 | 0.6325 | vi | Southwest Airlines Co. (NYSE: LUV) là một hãng hàng không lớn của Hoa Kỳ và là hãng hàng không giá rẻ lớn nhất thế giới, có trụ sở chính tại Dallas, Texas. Hãng hàng không được ... |
| 7 | 4737106 | 0 | 0.6322 | vi | Hãng hàng không EVA Air có trụ sở tại Đài Loan đã tung ra một chiếc máy bay có chủ đề Hello Kitty cho các tuyến Los Angeles và Paris đến Đài Loan, và những chiếc máy bay này sẽ ... |
| 8 | 4653036 | 0 | 0.6291 | vi | Houston cũng là trụ sở chính của United Airlines, và Bush Intercontinental là trung tâm lớn nhất của United, với 800 chuyến khởi hành hàng ngày. Sân bay Liên lục địa George Bush... |
| 9 | 8576327 | 0 | 0.6288 | vi | Số phiếu tự tin 298K. Southwest Airlines hiện là hãng hàng không lớn nhất bay từ Sân bay Hobby ở Houston, Texas. Delta Air Lines, American Airlines, Air Tran Airlines và JetBlue... |
| 10 | 820650 | 0 | 0.6286 | vi | American Airlines, Inc., hoạt động với tên gọi American Airlines (AA), là một hãng hàng không của Hoa Kỳ và là công ty con của AMR Corporation. Có trụ sở chính tại Fort Worth, T... |

Mixed top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 7422184 | 1 | 0.6761 | vi | HÔM NAY TRÊN TRỜI: Spirit Airlines tăng phí hành lý ký gửi cho các ngày lễ. Trong số 10 tuyến đường mới mà Spirit có kế hoạch đến Houston, ba tuyến đến Mexico và bốn tuyến đến T... |
| 2 | 7422182 | 0 | 0.6565 | vi | Tinh thần: Việc mở rộng quy mô làm cho Houston trở thành cơ sở quốc tế. Hãng hàng không Spirit Airlines thông báo về việc mở rộng lớn tại Liên lục địa Houston Bush, cho biết họ ... |
| 3 | 6494139 | 0 | 0.6559 | vi | Spirit khai thác các chuyến bay theo lịch trình trên khắp Hoa Kỳ và Caribe, Mexico, Mỹ Latinh và Nam Mỹ. Hãng hàng không khai thác các căn cứ tại Atlantic City, ChicagoÃƒÂ ¢ Ã‚â... |
| 4 | 4040132 | 0 | 0.6526 | vi | Spirit Airlines, Inc. (NASDAQ: SAVE) là một hãng hàng không giá cực rẻ của Mỹ, có trụ sở chính tại Miramar, Florida. Spirit khai thác các chuyến bay theo lịch trình trên khắp Ho... |
| 5 | 5371608 | 0 | 0.6496 | vi | Spirit khai thác các chuyến bay theo lịch trình trên khắp Hoa Kỳ cũng như Caribe, Mexico và Mỹ Latinh. Các thành phố trọng điểm chính bao gồm; Ft. Lauderdale, Dallas / Fort Wort... |
| 6 | 1499367 | 0 | 0.6331 | vi | Liên hệ với Trung tâm Dịch vụ Khách hàng của Spirit Airlines. Spirit Air là một trong những hãng hàng không ÃƒÂ ¢ Ã‚â‚¬Ã‚Å “BudgetÃƒÂ ¢ Ã‚â‚¬Ã‚ hàng đầu. Được thành lập vào năm ... |
| 7 | 5858258 | 0 | 0.6279 | vi | Kiểm tra các chuyến bay của Spirit Airlines đến Houston Las Vegas, tìm kiếm vé máy bay giá rẻ và đặt vé trong tích tắc! Hãng hàng không Spirit Airlines có 281 chuyến bay Spirit ... |
| 8 | 5858259 | 0 | 0.6243 | vi | Lịch bay của Spirit Airlines đến Houston Las Vegas. Nhận tổng quan nhanh về các chuyến bay đến và đi từ các điểm đến mà bạn đang tìm kiếm. mang đến cho bạn lịch trình bay của Hã... |
| 9 | 8576327 | 0 | 0.6210 | vi | Số phiếu tự tin 298K. Southwest Airlines hiện là hãng hàng không lớn nhất bay từ Sân bay Hobby ở Houston, Texas. Delta Air Lines, American Airlines, Air Tran Airlines và JetBlue... |
| 10 | 8576325 | 0 | 0.6191 | vi | tồn tại và là một thay thế của. Southwest Airlines hiện là hãng hàng không lớn nhất bay từ Sân bay Hobby ở Houston, Texas. Delta Air Lines, American Airlines, Air Tran Airlines ... |

### qid `880766`

- query A (`en`): what nationality is male name ish
- query B (`vi`): quốc tịch nam là gì
- diagnosis: Unclassified; nDCG@10 end=0.0000, mix=100.0000, Δ=100.0000; Recall@10 end=0.0000, mix=100.0000, Δ=100.0000; tokens(a/b)=6/5, len_ratio=1.2000; overlap@10=0; source=evaluate_perquery; focus=best (gain)

Endpoint top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 6373404 | 0 | 0.5778 | vi | Quốc tịch kép. Mục 101 (a) (22) của Đạo luật Nhập cư và Quốc tịch (INA) quy định rằng ÃƒÂ ¢ Ã‚â‚¬Ã‚Å “thuật ngữ ÃƒÂ ¢ Ã‚â‚¬Ã‚Ëœ quốc tịch của Hoa KỳÃƒÂ ¢ Ã‚â ‚¬Ã‚â„ ¢ có nghĩa l... |
| 2 | 4412903 | 0 | 0.5697 | vi | 1 Từ này thường dùng để chỉ các công dân của Hợp chủng quốc Hoa Kỳ chứ không phải những người sống ở Nam Mỹ. Công dân Nam Mỹ thường được gọi bằng cách sử dụng các tính từ lấy từ... |
| 3 | 7305406 | 0 | 0.5640 | vi | Chúng ta chỉ bắt đầu trở thành nam giới sau khi đã trở thành công dân .. Quyền công dân thường được chứng minh bằng hộ chiếu do nhà nước cấp. Quốc tịch nghĩa vụ cá nhân đối với ... |
| 4 | 7437328 | 0 | 0.5617 | vi | Quốc tịch Anh (Ở nước ngoài) Quốc tịch Anh (Ở nước ngoài), thường được gọi là BN (O), là một trong những nhóm quốc tịch Anh chính theo luật quốc tịch Anh. Người có quốc tịch này... |
| 5 | 1316533 | 0 | 0.5585 | vi | quốc tịch là quốc gia bạn sinh ra thuộc sắc tộc bao gồm bất cứ điều gì từ chủng tộc ngôn ngữ tôn giáo, phong tục và tôn giáo khiến bạn thuộc về một nhóm nhất định, ví dụ một ngư... |
| 6 | 4286249 | 0 | 0.5581 | vi | Quốc tịch là một mối quan hệ pháp lý giữa một cá nhân và một nhà nước. [1] Quốc tịch trao quyền tài phán của nhà nước đối với người đó và dành cho người đó sự bảo vệ của nhà nướ... |
| 7 | 7926653 | 0 | 0.5552 | vi | Quốc tịch là quốc gia bạn sinh ra, bất kể bạn có quốc tịch hay không. Mỹ không phải là một quốc gia. Hợp chủng quốc Hoa Kỳ là. Vì vậy, quốc tịch của bạn là Hợp chủng quốc Hoa Kỳ... |
| 8 | 7829822 | 0 | 0.5552 | vi | Quốc tịch của bạn là quốc gia bạn đến: Mỹ, Canada và Nga đều là quốc tịch. Mỗi người đều có giới tính, chủng tộc, khuynh hướng tình dục ... và quốc tịch. Quốc tịch của một người... |
| 9 | 3978773 | 0 | 0.5552 | vi | Quốc tịch là tư cách của một người được công nhận theo phong tục hoặc luật pháp như là một thành viên hợp pháp của một quốc gia có chủ quyền hoặc một phần của một quốc gia. Một ... |
| 10 | 3149294 | 0 | 0.5541 | vi | Mục 101 (a) (22) của Đạo luật Nhập cư và Quốc tịch (INA) quy định rằng ÃƒÂ ¢ Ã‚â‚¬Ã‚Å “thuật ngữ ÃƒÂ ¢ Ã‚â‚¬Ã‚Ëœ quốc tịch của Hoa KỳÃƒÂ ¢ Ã‚â ‚¬Ã‚â„ ¢ có nghĩa là (A) một công ... |

Mixed top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 7874197 | 1 | 0.6214 | vi | Tên: Ish. Nam giới. Cách sử dụng: Ish là một cái tên phổ biến. Nó thường được sử dụng như một tên con trai (nam). Những người có tên Ish nói chung có nguồn gốc từ Hợp chủng quốc... |
| 2 | 79950 | 0 | 0.5812 | vi | Nam giới. Cách sử dụng: Ishmael, có nguồn gốc từ tiếng Do Thái, là một cái tên phổ biến. Nó thường được sử dụng như một tên con trai (nam). Những người có tên Ishmael nói chung ... |
| 3 | 1830343 | 0 | 0.5804 | vi | Nam giới. Cách sử dụng: Shane, có nguồn gốc từ tiếng Do Thái, là một cái tên rất phổ biến. Nó thường được sử dụng như một tên con trai (nam). Những người có tên Shane nói chung ... |
| 4 | 6916276 | 0 | 0.5803 | vi | Nam giới. Cách sử dụng: Ian, có nguồn gốc từ tiếng Do Thái, là một cái tên rất phổ biến. Nó thường được sử dụng như một tên con trai (nam). Những người có tên Ian nói chung có n... |
| 5 | 6468167 | 0 | 0.5731 | vi | Nam giới. Cách sử dụng: Hamish, có nguồn gốc từ tiếng Do Thái, là một cái tên phổ biến. Nó thường được sử dụng như một tên con trai (nam). Những người có tên Hamish nói chung có... |
| 6 | 4577044 | 0 | 0.5719 | vi | Dân tộc: Mẹ anh là người Đức và bố anh là người Ireland. Anh ấy tự miêu tả mình có nền tảng tiếng Pháp bằng cách thay đổi cách phát âm họ của anh ấy cho chương trình của anh ấy,... |
| 7 | 1164961 | 0 | 0.5642 | vi | Tên: Ike. Nam giới. Cách sử dụng: Ike, có nguồn gốc từ tiếng Do Thái, là một cái tên phổ biến. Nó thường được sử dụng như một tên con trai (nam). Những người có tên Ike nói chun... |
| 8 | 5385040 | 0 | 0.5631 | vi | Giới tính: Unisex (Nam và Nữ) Cách sử dụng: Isha, có nguồn gốc từ tiếng Do Thái, là một cái tên phổ biến. Nó thường được sử dụng như một tên unisex (nam và nữ). Những người có t... |
| 9 | 8486991 | 0 | 0.5597 | vi | Nam giới. Cách sử dụng: James, có nguồn gốc từ tiếng Do Thái, là một cái tên rất phổ biến. Nó thường được sử dụng như một tên con trai (nam). Những người có tên James nói chung ... |
| 10 | 5454978 | 0 | 0.5591 | vi | Nam giới. Cách sử dụng: Ian, có nguồn gốc từ tiếng Do Thái, là một cái tên rất phổ biến. Nó thường được sử dụng như một tên con trai (nam). Những người có tên Ian nói chung có n... |

### qid `927196`

- query A (`en`): what year did they start to require act exam for college admissions
- query B (`vi`): năm nào họ bắt đầu yêu cầu thi hành vi để được tuyển sinh đại học
- diagnosis: Unclassified; nDCG@10 end=0.0000, mix=100.0000, Δ=100.0000; Recall@10 end=0.0000, mix=100.0000, Δ=100.0000; tokens(a/b)=12/16, len_ratio=0.7500; overlap@10=0; source=evaluate_perquery; focus=best (gain)

Endpoint top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 4911015 | 0 | 0.6063 | vi | Năm 1972, Đạo luật Cơ hội Việc làm Bình đẳng đã được thông qua, đặt cơ sở cho hành động khẳng định. Năm 1978, Tòa án Tối cao ra phán quyết trong vụ kiện Bakke kiện Univ. của Cal... |
| 2 | 2895053 | 0 | 0.6015 | vi | Vào tháng 5, trường Đại học đã công bố những thay đổi đối với tiêu chuẩn tuyển sinh chú trọng hơn vào thành tích học tập và lần đầu tiên đồng ý công khai quy trình tuyển sinh củ... |
| 3 | 7350458 | 0 | 0.5947 | vi | Thực hành này bắt đầu phổ biến tại các trường cao đẳng và đại học công lập vào những năm 1990, và hiện nay một số cơ sở và hệ thống ở hầu hết các bang đều yêu cầu nó. Đối với nh... |
| 4 | 2570605 | 0 | 0.5899 | vi | Theo Teachout, sự hiểu biết của chúng tôi về vận động hành lang và Tu chính án thứ nhất đã trải qua một quá trình xem xét lại tương tự, nếu lâu hơn và ít được dàn dựng một cách ... |
| 5 | 7801342 | 0 | 0.5877 | vi | Thử nghiệm xảy ra trong một môi trường được kiểm soát chặt chẽ. Năm 1979, Cơ quan Lập pháp Texas quyết định rằng tất cả học sinh trong các trường công lập phải làm một bài kiểm ... |
| 6 | 3529471 | 0 | 0.5871 | vi | 1917Congress thông qua Đạo luật nghĩa vụ tuyển chọn vào ngày 18 tháng 5 năm 1917. Tất nhiên, đạo luật này yêu cầu tất cả nam giới từ 18-25 tuổi phải đăng ký tham gia quân dịch. ... |
| 7 | 6037153 | 0 | 0.5852 | vi | Texas House Bill 588, thường được gọi là Quy tắc 10% hàng đầu, là một đạo luật của Texas được thông qua vào năm 1997. Luật đảm bảo cho học sinh Texas tốt nghiệp trong mười phần ... |
| 8 | 5227548 | 0 | 0.5847 | vi | Hoa Kỳ lần đầu tiên thông qua quy định thời bình với Đạo luật đào tạo và phục vụ có chọn lọc năm 1940. Đạo luật này quy định rằng không quá 900.000 nam giới phải được đào tạo cù... |
| 9 | 1447243 | 0 | 0.5829 | vi | Những thay đổi về tính đủ điều kiện ban đầu sẽ bắt đầu vào mùa hè năm 2015 và sẽ bắt buộc đối với tất cả sinh viên năm nhất đại học mới nhập học kể từ thời điểm đó. Gần đây, Ủy ... |
| 10 | 2398281 | 0 | 0.5828 | vi | Mặc dù bài kiểm tra do quân đội quản lý, nhưng không phải (và chưa bao giờ) yêu cầu người dự thi đạt điểm đủ tiêu chuẩn phải gia nhập lực lượng vũ trang. ASVAB được giới thiệu l... |

Mixed top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 7735310 | 1 | 0.6769 | vi | ACT (/ eÃƒâ € ° Ã‚Âª s iÃƒâ € ¹Ã ‚t iÃƒâ € ¹Ã‚ /; ban đầu là tên viết tắt của American College Testing) là một bài kiểm tra tiêu chuẩn được sử dụng để tuyển sinh đại học ở Hoa K... |
| 2 | 5181020 | 0 | 0.6551 | vi | Việc sử dụng ACT của các trường đại học đã tăng lên do nhiều lời chỉ trích về tính hiệu quả và công bằng của kỳ thi SAT. American Mensa là một xã hội có chỉ số IQ cao cho phép s... |
| 3 | 3169887 | 0 | 0.6477 | vi | Nó được tổ chức lần đầu tiên vào tháng 11 năm 1959 bởi Everett Franklin Lindquist với tư cách là đối thủ của Bài kiểm tra Năng lực Học vấn của College Board, nay là SAT. ACT ban... |
| 4 | 6168921 | 0 | 0.6384 | vi | Ở Mississippi, điểm thi ACT từ lâu đã được sử dụng như một công cụ phân biệt trong các trường đại học và cao đẳng tiểu bang. Năm 1962, hội đồng giáo dục đại học Mississippi đưa ... |
| 5 | 7612571 | 0 | 0.6329 | vi | (Cái còn lại là ACT.) Nó được điều hành bởi College Board, một tổ chức phi lợi nhuận cũng quản lý chương trình PSAT và AP (Advanced Placement). SAT ban đầu được chuyển thể từ và... |
| 6 | 4315626 | 0 | 0.6321 | vi | 1 2005: ACT bổ sung một Bài kiểm tra Viết tùy chọn. 2 2007: Mọi trường cao đẳng ở Hoa Kỳ hiện chấp nhận ACT để nhập học. 3 2012: Lần đầu tiên số học sinh thi ACT vượt qua SAT. V... |
| 7 | 2145426 | 0 | 0.6310 | vi | Số học sinh thi ACT trong năm học 1967-68 đạt khoảng 950.000, gấp hơn bảy lần số học sinh thi ACT năm 1959-60, năm thi ACT đầu tiên. Đại học California bắt đầu yêu cầu ứng viên ... |
| 8 | 3557078 | 0 | 0.6304 | vi | ÃƒÂ ¢ Ã‚â‚¬Ã‚Â ¢ Ở Kentucky, luật tiểu bang đã yêu cầu các trường cấp ACT cho học sinh kể từ năm 2008, và tỷ lệ đậu đại học từ đó đã được cải thiện. Kết quả được sử dụng cho mục... |
| 9 | 6493307 | 0 | 0.6274 | vi | Thi SAT hoặc đối thủ cạnh tranh của nó, ACT, là bắt buộc đối với sinh viên năm nhất vào nhiều, nhưng không phải tất cả, các trường đại học ở Hoa Kỳ. Vào ngày 5 tháng 3 năm 2014,... |
| 10 | 1963238 | 0 | 0.6261 | vi | Penn thông báo các yêu cầu kiểm tra mới cho các ứng viên. Thứ Sáu, ngày 31 tháng 7 năm 2015. Bắt đầu với chu kỳ tuyển sinh 2015-2016, Đại học Pennsylvania sẽ yêu cầu tất cả các ... |

### qid `333327`

- query A (`en`): how old do you have to be in a wisconsin casino
- query B (`vi`): bạn phải bao nhiêu tuổi để có mặt trong sòng bạc khôn ngoan
- diagnosis: Unclassified; nDCG@10 end=30.1030, mix=100.0000, Δ=69.8970; Recall@10 end=100.0000, mix=100.0000, Δ=0.0000; tokens(a/b)=11/13, len_ratio=0.8462; overlap@10=5; source=evaluate_perquery; focus=best (gain)

Endpoint top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 486429 | 0 | 0.7144 | vi | Đánh giá Mới nhất Cũ nhất. Câu trả lời hay nhất: Độ tuổi vào sòng bạc là 21 nhưng bạn luôn có thể lạnh sống lưng trong khu vực khách sạn. Bạn có thể cân nhắc đến Oklahoma vì độ ... |
| 2 | 8457141 | 0 | 0.7073 | vi | Để vào sòng bạc, bạn phải từ 19 tuổi trở lên. Bảo vệ sẽ yêu cầu ID từ những khách hàng dưới 30 tuổi. Khách hàng từ 21 tuổi trở xuống sẽ được yêu cầu xuất trình hai mảnh giấy tờ ... |
| 3 | 789945 | 0 | 0.7034 | vi | Khách phải từ 18 tuổi trở lên mới được chơi trên bàn và máy đánh bạc. Khách dưới 18 tuổi không được phép vào sòng bạc. Khách phải từ 18 tuổi trở lên mới có thể mua thẻ lô tô và ... |
| 4 | 6514436 | 0 | 0.7025 | vi | Tuổi đánh bạc hợp pháp trong sòng bạc của bạn là bao nhiêu? Độ tuổi hợp pháp để chơi Trò chơi trên bàn, Slots, Race Book và Keno là 21. * Vào ngày bạn bước sang tuổi 21, bạn sẽ ... |
| 5 | 7336480 | 0 | 0.6990 | vi | Bạn thực sự có thể vào sòng bạc / khách sạn ở bất kỳ độ tuổi nào nhưng bạn phải 18 tuổi trong một cuộc chơi và 21 tuổi để thực sự đánh bạc. Tùy thuộc vào, Một số sòng bạc ID bạn... |
| 6 | 7336481 | 0 | 0.6989 | vi | Câu trả lời hay nhất: Nó thực sự phụ thuộc vào vị trí của sòng bạc và bạn định làm gì khi ở đó. Một số sòng bạc có và giới hạn độ tuổi là 18, một số 19, một số 21. Một số sòng b... |
| 7 | 7194716 | 0 | 0.6988 | vi | Khách phải từ 18 tuổi trở lên mới được chơi trên bàn và máy đánh bạc. Khách dưới 18 tuổi không được phép vào sòng bạc. Khách phải từ 18 tuổi trở lên mới có thể mua thẻ lô tô và ... |
| 8 | 7486777 | 0 | 0.6978 | vi | Độ tuổi đánh bạc tối thiểu tại hầu hết các sòng bạc (tại một số sòng bạc là 18) và 18 đối với cá cược bingo hoặc pari-mutuel. Tìm trong danh sách ÃƒÂ ¢ Ã‚â‚¬Ã‚Å “Tính năng đặc b... |
| 9 | 7336477 | 1 | 0.6972 | vi | Bạn phải từ 21 tuổi trở lên để đến bất kỳ sòng bạc nào ở Hoa Kỳ, bao gồm cả bang Wisconsin. ChaCha vào! Bạn chỉ phải 18 tuổi để chơi bingo tại Sòng bạc Ho Chunk Wisconsin Dells ... |
| 10 | 6416723 | 0 | 0.6963 | vi | Bất cứ ai dự định đến một sòng bạc nên lưu ý rằng mặc dù bạn chỉ cần đủ 18 tuổi để đánh bạc, nhưng bạn phải đủ 21 tuổi để vào một quán bar. Nhiều sòng bạc cũng là quán bar phục ... |

Mixed top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 7336477 | 1 | 0.7816 | vi | Bạn phải từ 21 tuổi trở lên để đến bất kỳ sòng bạc nào ở Hoa Kỳ, bao gồm cả bang Wisconsin. ChaCha vào! Bạn chỉ phải 18 tuổi để chơi bingo tại Sòng bạc Ho Chunk Wisconsin Dells ... |
| 2 | 7336482 | 0 | 0.7720 | vi | Các câu trả lời. 1 Bạn phải từ 21 tuổi trở lên để đến bất kỳ sòng bạc nào ở Hoa Kỳ, kể cả tiểu bang Wisconsin. 2 Bạn chỉ phải 18 tuổi để chơi bingo tại Sòng bạc Ho Chunk Wiscons... |
| 3 | 7336478 | 0 | 0.7286 | vi | Những người đánh bạc ở Wisconsin sẽ có quyền truy cập vào cờ bạc hợp pháp ở tuổi 18. Bạn sẽ có thể tham gia ngay khi đến tuổi trưởng thành này. Hầu hết mọi loại cờ bạc đều có th... |
| 4 | 7336484 | 0 | 0.7278 | vi | Tôi không nghĩ rằng 18 tuổi có thể đến sòng bạc ở Wisconsin .... Tôi không nghĩ rằng 18 tuổi có thể đến sòng bạc ở Wisconsin :) Nếu bạn từ 21 tuổi trở lên, hãy đợi vài năm nữa v... |
| 5 | 486429 | 0 | 0.7276 | vi | Đánh giá Mới nhất Cũ nhất. Câu trả lời hay nhất: Độ tuổi vào sòng bạc là 21 nhưng bạn luôn có thể lạnh sống lưng trong khu vực khách sạn. Bạn có thể cân nhắc đến Oklahoma vì độ ... |
| 6 | 7336485 | 0 | 0.7253 | vi | Definetly nếu bạn muốn đến sòng bạc ở minnesota, bạn phải đủ 21 tuổi. Đi đến các quốc gia khác nếu bạn muốn đến sòng bạc và bạn còn quá trẻ nhưng tôi đề nghị bạn - đừng làm điều... |
| 7 | 6416728 | 0 | 0.7199 | vi | Tuổi cờ bạc ở Pennsylvania có vẻ hơi phức tạp. Tùy thuộc vào thứ bạn muốn chơi và nơi bạn muốn chơi, bạn cần phải 18 hoặc 21. Sòng bạc trên đất liền yêu cầu người đánh bạc phải ... |
| 8 | 8457141 | 0 | 0.7168 | vi | Để vào sòng bạc, bạn phải từ 19 tuổi trở lên. Bảo vệ sẽ yêu cầu ID từ những khách hàng dưới 30 tuổi. Khách hàng từ 21 tuổi trở xuống sẽ được yêu cầu xuất trình hai mảnh giấy tờ ... |
| 9 | 7336480 | 0 | 0.7138 | vi | Bạn thực sự có thể vào sòng bạc / khách sạn ở bất kỳ độ tuổi nào nhưng bạn phải 18 tuổi trong một cuộc chơi và 21 tuổi để thực sự đánh bạc. Tùy thuộc vào, Một số sòng bạc ID bạn... |
| 10 | 7194716 | 0 | 0.7120 | vi | Khách phải từ 18 tuổi trở lên mới được chơi trên bàn và máy đánh bạc. Khách dưới 18 tuổi không được phép vào sòng bạc. Khách phải từ 18 tuổi trở lên mới có thể mua thẻ lô tô và ... |

### qid `1099670`

- query A (`en`): how big can leopard tortoises get
- query B (`vi`): rùa báo có thể lớn như thế nào
- diagnosis: Unclassified; nDCG@10 end=31.5465, mix=100.0000, Δ=68.4535; Recall@10 end=100.0000, mix=100.0000, Δ=0.0000; tokens(a/b)=6/8, len_ratio=0.7500; overlap@10=7; source=evaluate_perquery; focus=best (gain)

Endpoint top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 7236804 | 0 | 0.7741 | vi | Câu trả lời hay nhất: Rùa báo là loài rùa lớn thứ tư trên thế giới, với những con trưởng thành điển hình dài tới 18 inch (460 mm) và nặng 40 pound (18 kg). Ví dụ lớn có thể dài ... |
| 2 | 1493058 | 0 | 0.7718 | vi | Rùa báo ăn thực vậtB al. Rùa báo là loài rùa lớn thứ tư trên thế giới, với những con trưởng thành điển hình dài tới 18 inch (460 mm) và nặng 40 pound (18 kg). Ví dụ lớn có thể d... |
| 3 | 7236808 | 0 | 0.7603 | vi | Rùa báo trưởng thành có chiều dài từ 10 đến 18 inch tùy thuộc vào nguồn gốc địa lý và phân loài của rùa. Các loài phụ ở Nam Phi, Stigmachelys pardalis pardalis, có thể phát triể... |
| 4 | 3211276 | 0 | 0.7523 | vi | Rùa da báo là loài rùa lớn thứ tư trên thế giới, với những con trưởng thành điển hình đạt 18 inch (460 mm) và nặng 40 pound (18 kg). Ví dụ lớn có thể dài 70 cm (28 in) và nặng t... |
| 5 | 3211275 | 0 | 0.7416 | vi | Rùa báo ăn thực vậtB al. Rùa báo là loài rùa lớn thứ tư trên thế giới, với những con trưởng thành điển hình dài tới 18 inch (460 mm) và nặng 40 pound (18 kg). Ví dụ lớn có thể d... |
| 6 | 6701274 | 0 | 0.7151 | vi | Rùa Báo, tên chính thức là Rùa Báo Châu Phi (Geochelone Pardalis) có thể dài tới khoảng 0,6 m và có thể nặng tới 70 lbs nếu được chăm sóc thích hợp (và có gen tốt). Chúng là vật... |
| 7 | 6492050 | 0 | 0.7001 | vi | Cách Chăm sóc Rùa Báo. Rùa Báo, chính thức được gọi là Rùa Báo Châu Phi (Geochelone Pardalis) có thể dài tới 2 feet (0,6 m) và có thể nặng tới 70 lbs nếu được chăm sóc thích hợp... |
| 8 | 7236803 | 1 | 0.6976 | vi | Tên cho Leopard Tortoises. Kích thước của Leopard Tortoises. Trung bình, rùa báo có chiều dài khoảng 10-18 inch (với một số loài con phát triển dài tới 30 inch) và nặng khoảng 4... |
| 9 | 5784536 | 0 | 0.6858 | vi | Một con báo trưởng thành nặng 80-140 pound, cao khoảng 32 inch ở vai và dài 48-56 inch từ đầu đến thân với 28-32 inch ở đuôi của con đực lớn hơn một chút so với con cái. |
| 10 | 5784534 | 0 | 0.6816 | vi | Con báo trưởng thành nặng từ 40 đến 65 kg (88 đến 140 lb). Tổng chiều dài cơ thể của nó từ 115 đến 135 cm (45 đến 53 in), trong khi đuôi có thể dài tới 84 cm (33 in). |

Mixed top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 7236803 | 1 | 0.7563 | vi | Tên cho Leopard Tortoises. Kích thước của Leopard Tortoises. Trung bình, rùa báo có chiều dài khoảng 10-18 inch (với một số loài con phát triển dài tới 30 inch) và nặng khoảng 4... |
| 2 | 7236808 | 0 | 0.7285 | vi | Rùa báo trưởng thành có chiều dài từ 10 đến 18 inch tùy thuộc vào nguồn gốc địa lý và phân loài của rùa. Các loài phụ ở Nam Phi, Stigmachelys pardalis pardalis, có thể phát triể... |
| 3 | 3211276 | 0 | 0.7257 | vi | Rùa da báo là loài rùa lớn thứ tư trên thế giới, với những con trưởng thành điển hình đạt 18 inch (460 mm) và nặng 40 pound (18 kg). Ví dụ lớn có thể dài 70 cm (28 in) và nặng t... |
| 4 | 7236804 | 0 | 0.7244 | vi | Câu trả lời hay nhất: Rùa báo là loài rùa lớn thứ tư trên thế giới, với những con trưởng thành điển hình dài tới 18 inch (460 mm) và nặng 40 pound (18 kg). Ví dụ lớn có thể dài ... |
| 5 | 1493058 | 0 | 0.7201 | vi | Rùa báo ăn thực vậtB al. Rùa báo là loài rùa lớn thứ tư trên thế giới, với những con trưởng thành điển hình dài tới 18 inch (460 mm) và nặng 40 pound (18 kg). Ví dụ lớn có thể d... |
| 6 | 3211275 | 0 | 0.7010 | vi | Rùa báo ăn thực vậtB al. Rùa báo là loài rùa lớn thứ tư trên thế giới, với những con trưởng thành điển hình dài tới 18 inch (460 mm) và nặng 40 pound (18 kg). Ví dụ lớn có thể d... |
| 7 | 5567232 | 0 | 0.6880 | vi | Giới thiệu về Galapagos Tortoise. Loài rùa lớn nhất là Rùa Galapagos. Chúng có thể phát triển đến kích thước lên đến 880 pound. Chúng cũng có thể dài hơn 6 feet. Chúng có thể có... |
| 8 | 7236812 | 0 | 0.6864 | vi | Leopard Tortoise Life Span. Rùa da báo sống từ 50 đến 100 năm trong tự nhiên. Leopard Tortoise Caging. Thiết lập ưa thích cho rùa báo trưởng thành là ở ngoài trời. Tuy nhiên, nế... |
| 9 | 6701274 | 0 | 0.6788 | vi | Rùa Báo, tên chính thức là Rùa Báo Châu Phi (Geochelone Pardalis) có thể dài tới khoảng 0,6 m và có thể nặng tới 70 lbs nếu được chăm sóc thích hợp (và có gen tốt). Chúng là vật... |
| 10 | 5771876 | 0 | 0.6778 | vi | Hải cẩu Well Leopard có thân hình thon dài, đầu và hàm lớn. Chúng cũng có màu xám với những đốm trắng. Chúng không có tai và có râu. Chúng được gọi là báo gấm ÃƒÂ ¢ Ã‚â‚¬Ã‚Â¦ al... |

### qid `264594`

- query A (`en`): how long is super bowl game
- query B (`vi`): trò chơi siêu bát kéo dài bao lâu
- diagnosis: Unclassified; nDCG@10 end=31.5465, mix=100.0000, Δ=68.4535; Recall@10 end=100.0000, mix=100.0000, Δ=0.0000; tokens(a/b)=6/8, len_ratio=0.7500; overlap@10=6; source=evaluate_perquery; focus=best (gain)

Endpoint top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 6507049 | 0 | 0.6387 | vi | trận đấu bóng rổ kéo dài bao lâu. |
| 2 | 3599985 | 0 | 0.6156 | vi | Thông thường khoảng 3 giờ. Siêu tô là khoảng 4 giờ. Thời gian chơi trung bình của một trò chơi là 12:36 phút. |
| 3 | 6507055 | 0 | 0.6155 | vi | thời lượng của một trận đấu bóng rổ là bao nhiêu. |
| 4 | 1344234 | 0 | 0.6105 | vi | Các trò chơi thường kéo dài từ 2 giờ đến 15 phút và 2 giờ 30 phút. |
| 5 | 6507051 | 0 | 0.6004 | vi | trận bóng rổ đại học kéo dài bao lâu. |
| 6 | 4146196 | 0 | 0.5966 | vi | Super Bowl diễn ra bao nhiêu giờ? Trò chơi chính nó là 60 phút giống như bất kỳ trò chơi nào. Tuy nhiên, các chương trình quảng cáo, chương trình giữa giờ nghỉ giải lao và thườn... |
| 7 | 4146194 | 0 | 0.5962 | vi | Trò chơi chính nó là 60 phút giống như bất kỳ trò chơi nào. Tuy nhiên, các quảng cáo, chương trình giữa giờ nghỉ giải lao và thường xuyên nghỉ giải lao để xoa dịu các nhà quảng ... |
| 8 | 7320812 | 1 | 0.5921 | vi | Super Bowl thường kéo dài bao lâu? Super Bowl thường kéo dài bốn giờ. Bản thân trò chơi diễn ra trong khoảng ba giờ rưỡi, với một chương trình 30 phút giữa hiệp được tích hợp sẵ... |
| 9 | 6677058 | 0 | 0.5895 | vi | Super Bowl thường kéo dài bốn giờ. Bản thân trò chơi mất khoảng ba giờ rưỡi, với một chương trình 30 phút giữa hiệp được tích hợp sẵn. |
| 10 | 6921282 | 0 | 0.5892 | vi | Trò chơi có bốn phần tư -12 phút. Trò chơi kéo dài hơn 2 giờ một chút, tùy thuộc vào thời gian chờ, thời gian chờ truyền hình, thời gian nghỉ giải lao, thay đổi quyền sở hữu, v.... |

Mixed top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 7320812 | 1 | 0.7245 | vi | Super Bowl thường kéo dài bao lâu? Super Bowl thường kéo dài bốn giờ. Bản thân trò chơi diễn ra trong khoảng ba giờ rưỡi, với một chương trình 30 phút giữa hiệp được tích hợp sẵ... |
| 2 | 6677058 | 0 | 0.7221 | vi | Super Bowl thường kéo dài bốn giờ. Bản thân trò chơi mất khoảng ba giờ rưỡi, với một chương trình 30 phút giữa hiệp được tích hợp sẵn. |
| 3 | 6507055 | 0 | 0.7196 | vi | thời lượng của một trận đấu bóng rổ là bao nhiêu. |
| 4 | 6507049 | 0 | 0.7180 | vi | trận đấu bóng rổ kéo dài bao lâu. |
| 5 | 4146196 | 0 | 0.7132 | vi | Super Bowl diễn ra bao nhiêu giờ? Trò chơi chính nó là 60 phút giống như bất kỳ trò chơi nào. Tuy nhiên, các chương trình quảng cáo, chương trình giữa giờ nghỉ giải lao và thườn... |
| 6 | 7728746 | 0 | 0.7076 | vi | Super Bowl thường kéo dài bao lâu? Một trận bóng đá truyền thống kéo dài khoảng 3 giờ. Tuy nhiên, Super Bowl kéo dài khoảng 4 giờ từ đầu đến cuối. Trò chơi dài hơn do thời lượng... |
| 7 | 6507051 | 0 | 0.6977 | vi | trận bóng rổ đại học kéo dài bao lâu. |
| 8 | 7455534 | 0 | 0.6971 | vi | Nó sẽ kéo dài trong bao lâu? Các trận Super Bowl nổi tiếng là dài và kéo dài trung bình ba giờ 44 phút, theo The Verge. Một trận đấu NFL bình thường kéo dài hơn ba giờ một chút,... |
| 9 | 3440548 | 0 | 0.6943 | vi | Tuy nhiên, một trận bóng đá bình thường không kéo dài đến gần 60 phút. Trò chơi dừng lại vì nhiều lý do. Có những khoảng nghỉ ngắn giữa mỗi quý, giúp tăng thêm thời gian. Đồng h... |
| 10 | 6013768 | 0 | 0.6909 | vi | Tuy nhiên, Super Bowl kéo dài khoảng 4 giờ từ đầu đến cuối. Trò chơi dài hơn do thời lượng chiếu kéo dài và tập trung vào quảng cáo và thời gian nghỉ thương mại. Một trận bóng đ... |

### qid `577167`

- query A (`en`): what are you supposed to wear to a wake
- query B (`vi`): bạn phải mặc gì khi thức dậy
- diagnosis: Unclassified; nDCG@10 end=33.3333, mix=100.0000, Δ=66.6667; Recall@10 end=100.0000, mix=100.0000, Δ=0.0000; tokens(a/b)=9/7, len_ratio=1.2857; overlap@10=7; source=evaluate_perquery; focus=best (gain)

Endpoint top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 2498678 | 0 | 0.6517 | vi | Bạn phải thức đến ít nhất 3 giờ sáng. Tức là 4 giờ chiều. Bạn không chỉ phải mặc váy / đeo thẻ / vest mà còn phải để bạn cùng phòng vào bếp nấu bánh mì kẹp thịt phô mai và uống ... |
| 2 | 7709488 | 0 | 0.6510 | vi | Bạn phải mặc gì khi thức dậy / xem đám tang? Tôi sẽ đi thăm các cụ bà của tôi, quảng cáo sau đó là đám tang của ông ấy, nhưng tôi thực sự không biết mình phải mặc gì !!!! bất cứ... |
| 3 | 7709485 | 0 | 0.6441 | vi | Có một số sự linh hoạt trong trang phục khi thức dậy dựa trên nơi tổ chức đánh thức, nhưng quy tắc chung về màu sắc là giữ cho các tông màu u ám như đen, xanh nước biển và tối h... |
| 4 | 2847327 | 0 | 0.6418 | vi | Những món đồ khác mà bạn kết hợp với áo khoác thể thao và quần jean sẽ giúp đảm bảo việc ngủ dậy của bạn đạt hiệu quả cao. Bạn có thể chọn cách ăn mặc lịch sự hơn một chút hoặc ... |
| 5 | 7709484 | 0 | 0.6404 | vi | thức: màu xám với màu đen tang lễ: màu đen với các phụ kiện màu trắng hoặc bạc Đừng quá lòe loẹt, một chiếc váy sẽ hoàn hảo cho đám tang và buổi thức. Mặc dù để thức dậy, bạn có... |
| 6 | 1699498 | 0 | 0.6390 | vi | Trang phục buổi sáng là quy định về trang phục ban ngày, chủ yếu dành cho nam giới với áo khoác buổi sáng, áo ghi lê và quần tây sọc, và một bộ váy thích hợp cho nữ giới. |
| 7 | 7709487 | 1 | 0.6386 | vi | Tốt nhất là nên tránh mặc trang trọng nếu không biết quy định về trang phục khi thức dậy. Quần jean đôi khi có thể được chấp nhận tùy thuộc vào người đã khuất và gia đình cũng n... |
| 8 | 5347448 | 0 | 0.6351 | vi | Đăt báo thức của bạn. Rất nhiều người thích đồng hồ báo thức radio. Luôn luôn tuyệt vời khi có bài hát yêu thích của bạn để đánh thức bạn vào buổi sáng. Rất nhiều người cần khoả... |
| 9 | 7037897 | 0 | 0.6287 | vi | Rất nhiều người thích đồng hồ báo thức radio. Luôn luôn tuyệt vời khi có bài hát yêu thích của bạn để đánh thức bạn vào buổi sáng. Rất nhiều người cần khoảng một giờ để đứng dậy... |
| 10 | 4779309 | 0 | 0.6257 | vi | bộ đồ ngủ của tôi, tôi vừa mới thức dậy. quần áo của tôi trong ngày! một chiếc áo phông hoặc áo ba lỗ xinh xắn với một chiếc váy xinh xắn và những đôi giày bệt hoặc giày cao gót... |

Mixed top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 7709487 | 1 | 0.6541 | vi | Tốt nhất là nên tránh mặc trang trọng nếu không biết quy định về trang phục khi thức dậy. Quần jean đôi khi có thể được chấp nhận tùy thuộc vào người đã khuất và gia đình cũng n... |
| 2 | 7709485 | 0 | 0.6507 | vi | Có một số sự linh hoạt trong trang phục khi thức dậy dựa trên nơi tổ chức đánh thức, nhưng quy tắc chung về màu sắc là giữ cho các tông màu u ám như đen, xanh nước biển và tối h... |
| 3 | 7709488 | 0 | 0.6501 | vi | Bạn phải mặc gì khi thức dậy / xem đám tang? Tôi sẽ đi thăm các cụ bà của tôi, quảng cáo sau đó là đám tang của ông ấy, nhưng tôi thực sự không biết mình phải mặc gì !!!! bất cứ... |
| 4 | 7709484 | 0 | 0.6466 | vi | thức: màu xám với màu đen tang lễ: màu đen với các phụ kiện màu trắng hoặc bạc Đừng quá lòe loẹt, một chiếc váy sẽ hoàn hảo cho đám tang và buổi thức. Mặc dù để thức dậy, bạn có... |
| 5 | 2498678 | 0 | 0.6461 | vi | Bạn phải thức đến ít nhất 3 giờ sáng. Tức là 4 giờ chiều. Bạn không chỉ phải mặc váy / đeo thẻ / vest mà còn phải để bạn cùng phòng vào bếp nấu bánh mì kẹp thịt phô mai và uống ... |
| 6 | 2847327 | 0 | 0.6429 | vi | Những món đồ khác mà bạn kết hợp với áo khoác thể thao và quần jean sẽ giúp đảm bảo việc ngủ dậy của bạn đạt hiệu quả cao. Bạn có thể chọn cách ăn mặc lịch sự hơn một chút hoặc ... |
| 7 | 1699498 | 0 | 0.6425 | vi | Trang phục buổi sáng là quy định về trang phục ban ngày, chủ yếu dành cho nam giới với áo khoác buổi sáng, áo ghi lê và quần tây sọc, và một bộ váy thích hợp cho nữ giới. |
| 8 | 1744603 | 0 | 0.6284 | vi | Bộ đồ buổi sáng bao gồm một chiếc áo khoác buổi sáng màu xám giữa với một chiếc áo khoác ghi lê và quần tây vừa vặn. và áo ghi lê phải cùng loại vải (hoặc ít nhất là cùng màu) v... |
| 9 | 4470761 | 0 | 0.6222 | vi | 1 Tuy nhiên, luôn có những ngoại lệ đối với các quy tắc. 2 Bộ vest màu xanh lam với áo cổ lọ màu đen, áo sơ mi đen với cà vạt dài màu đỏ, áo sơ mi đen (không cà vạt; không cài c... |
| 10 | 4470754 | 0 | 0.6210 | vi | 1 Nên mặc gì nếu bạn đang tham dự lễ báo thức, đám tang và lễ viếng mộ: khi thức dậy bạn có thể mặc chủ yếu là màu đen với một chút màu sắc. 2 Trẻ em thường có thể mặc giống nha... |

### qid `988306`

- query A (`en`): who owns charlotte hilton university place hotel
- query B (`vi`): người sở hữu khách sạn đại học charlotte hilton
- diagnosis: Unclassified; nDCG@10 end=33.3333, mix=100.0000, Δ=66.6667; Recall@10 end=100.0000, mix=100.0000, Δ=0.0000; tokens(a/b)=7/10, len_ratio=0.7000; overlap@10=9; source=evaluate_perquery; focus=best (gain)

Endpoint top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 1221518 | 0 | 0.6677 | vi | Hilton Inc. (trước đây là Hilton Worldwide Holdings, Inc. và Tập đoàn Khách sạn Hilton) là một công ty khách sạn đa quốc gia của Mỹ quản lý và nhượng quyền một loạt các khách sạ... |
| 2 | 4138386 | 0 | 0.6638 | vi | Tập đoàn khách sạn Hilton thuộc sở hữu của Tập đoàn Blackstone. Công ty từng thuộc sở hữu của các cổ đông, hầu hết là hậu duệ của người sáng lập công ty, Conrad Hilton. Tập đoàn... |
| 3 | 4138393 | 0 | 0.6622 | vi | Hilton Worldwide Holdings, Inc. (trước đây là Hilton Worldwide và Hilton Hotels Corporation) là một công ty khách sạn đa quốc gia của Mỹ quản lý và nhượng quyền một loạt các khá... |
| 4 | 4571271 | 0 | 0.6501 | vi | Tập đoàn khách sạn Hilton. Tập đoàn Khách sạn Hilton của Hoa Kỳ là công ty khách sạn hàng đầu sở hữu, quản lý và nhượng quyền hơn 2.000 khách sạn trên khắp đất nước. Chi nhánh q... |
| 5 | 565367 | 0 | 0.6457 | vi | Chủ sở hữu ban đầu là các doanh nhân Hyatt Robert von Dehn và Jack Dyer Crouch; sau một vài năm, Von Dehn bán cổ phần của mình trong khách sạn cho doanh nhân Jay Pritzker. Em tr... |
| 6 | 1510010 | 0 | 0.6423 | vi | Công ty ban đầu được thành lập bởi Conrad Hilton. Tính đến năm 2017, đã có hơn 570 khách sạn Hilton tại 85 quốc gia và vùng lãnh thổ trên sáu lục địa. Các tài sản thuộc sở hữu, ... |
| 7 | 7289333 | 1 | 0.6375 | vi | Hilton Charlotte University Place thuộc sở hữu của UPH Lakeside, LP, và được quản lý bởi GF Management, LLC, một trong những công ty quản lý khách sạn hàng đầu của quốc gia có t... |
| 8 | 7808101 | 0 | 0.6358 | vi | Ritz-Carlton Chicago để bán Các nhà đầu tư săn lùng các khách sạn sang trọng có cơ hội mua được một trong những bất động sản nổi tiếng nhất của thành phố: Ritz-Carlton Chicago. ... |
| 9 | 4138388 | 0 | 0.6353 | vi | Câu trả lời hay nhất: Mr Hilton - Bố của Paris !! Tập đoàn khách sạn Hilton thuộc sở hữu của Tập đoàn Blackstone. Công ty từng thuộc sở hữu của các cổ đông, hầu hết là hậu duệ c... |
| 10 | 7808104 | 0 | 0.6276 | vi | Cần bán Ritz-Carlton Chicago. Các nhà đầu tư săn lùng các khách sạn sang trọng có cơ hội mua một trong những bất động sản nổi tiếng nhất của thành phố: Ritz-Carlton Chicago. JMB... |

Mixed top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 7289333 | 1 | 0.6679 | vi | Hilton Charlotte University Place thuộc sở hữu của UPH Lakeside, LP, và được quản lý bởi GF Management, LLC, một trong những công ty quản lý khách sạn hàng đầu của quốc gia có t... |
| 2 | 1221518 | 0 | 0.6514 | vi | Hilton Inc. (trước đây là Hilton Worldwide Holdings, Inc. và Tập đoàn Khách sạn Hilton) là một công ty khách sạn đa quốc gia của Mỹ quản lý và nhượng quyền một loạt các khách sạ... |
| 3 | 4138393 | 0 | 0.6464 | vi | Hilton Worldwide Holdings, Inc. (trước đây là Hilton Worldwide và Hilton Hotels Corporation) là một công ty khách sạn đa quốc gia của Mỹ quản lý và nhượng quyền một loạt các khá... |
| 4 | 4571271 | 0 | 0.6432 | vi | Tập đoàn khách sạn Hilton. Tập đoàn Khách sạn Hilton của Hoa Kỳ là công ty khách sạn hàng đầu sở hữu, quản lý và nhượng quyền hơn 2.000 khách sạn trên khắp đất nước. Chi nhánh q... |
| 5 | 4138386 | 0 | 0.6425 | vi | Tập đoàn khách sạn Hilton thuộc sở hữu của Tập đoàn Blackstone. Công ty từng thuộc sở hữu của các cổ đông, hầu hết là hậu duệ của người sáng lập công ty, Conrad Hilton. Tập đoàn... |
| 6 | 565367 | 0 | 0.6382 | vi | Chủ sở hữu ban đầu là các doanh nhân Hyatt Robert von Dehn và Jack Dyer Crouch; sau một vài năm, Von Dehn bán cổ phần của mình trong khách sạn cho doanh nhân Jay Pritzker. Em tr... |
| 7 | 7808101 | 0 | 0.6311 | vi | Ritz-Carlton Chicago để bán Các nhà đầu tư săn lùng các khách sạn sang trọng có cơ hội mua được một trong những bất động sản nổi tiếng nhất của thành phố: Ritz-Carlton Chicago. ... |
| 8 | 1510010 | 0 | 0.6299 | vi | Công ty ban đầu được thành lập bởi Conrad Hilton. Tính đến năm 2017, đã có hơn 570 khách sạn Hilton tại 85 quốc gia và vùng lãnh thổ trên sáu lục địa. Các tài sản thuộc sở hữu, ... |
| 9 | 90721 | 0 | 0.6244 | vi | Chủ sở hữu mới cho khách sạn Crowne Plaza ở Clayton. MỚI CROWNE: DeVault Investments thuộc sở hữu địa phương đã mua khách sạn Crowne Plaza 250 phòng ở Clayton. DeVault, có trụ s... |
| 10 | 7808104 | 0 | 0.6233 | vi | Cần bán Ritz-Carlton Chicago. Các nhà đầu tư săn lùng các khách sạn sang trọng có cơ hội mua một trong những bất động sản nổi tiếng nhất của thành phố: Ritz-Carlton Chicago. JMB... |

### qid `1051755`

- query A (`en`): who sings take this world by storm
- query B (`vi`): ai hát đưa thế giới này đi trong cơn bão
- diagnosis: Unclassified; nDCG@10 end=35.6207, mix=100.0000, Δ=64.3793; Recall@10 end=100.0000, mix=100.0000, Δ=0.0000; tokens(a/b)=7/10, len_ratio=0.7000; overlap@10=7; source=evaluate_perquery; focus=best (gain)

Endpoint top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 1681605 | 0 | 0.5325 | vi | Lưu trữ web với TotalChoice Nhấp và bắt đầu kiếm tiền! Chính sách bảo mật. ĐẾN BÃO BÃO BẰNG (Mosie Lister) Người ghi: Bill Gaither; Vestal Goodman; Tầm nhìn xa hơn; Nguồn cảm hứ... |
| 2 | 2544736 | 0 | 0.5202 | vi | Làm cho thế giới đi xa. Từ Wikipedia, bách khoa toàn thư miễn phí. Make the World Go Away 'là một ca khúc nhạc đồng quê nổi tiếng do Hank Cochran sáng tác. Nó đã trở thành Top 4... |
| 3 | 6361208 | 0 | 0.5173 | vi | Và nó cảm thấy, yeah nó giống như. Thế giới đã trở nên lạnh giá. Bây giờ bạn đã đi xa. Đi đi, biến đi. Vâng, vâng, vâng, vâng, vâng. Đi đi, biến đi. Nhạc sĩ. BRYAN HOLLAND. Xuất... |
| 4 | 2672795 | 0 | 0.5165 | vi | Lời dẫn đó và câu chuyện khô khan của tạp chí tin tức đã truyền cảm hứng cho Gordon Lightfoot viết nên một trong những bài hát về câu chuyện hay nhất từ ​​trước đến nay. Vào ngà... |
| 5 | 7910190 | 0 | 0.5144 | vi | Lời dẫn đó và câu chuyện khô khan của tạp chí tin tức đã truyền cảm hứng cho Gordon Lightfoot viết nên một trong những bài hát về câu chuyện hay nhất từ ​​trước đến nay. Vào ngà... |
| 6 | 7928929 | 1 | 0.5067 | vi | Giới thiệu về ÃƒÂ ¢ Ã‚â‚¬Ã‚Å “Take the World By StormÃƒÂ ¢ Ã‚â‚¬Ã‚. ÃƒÂ ¢ Ã‚â‚¬Ã‚Å “Take the World của StormÃƒÂ ¢ Ã‚â‚¬Ã‚ là một bài hát của ban nhạc soul-pop Đan Mạch Lukas Gra... |
| 7 | 1681604 | 0 | 0.5047 | vi | Trong bóng tối của nửa đêm, tôi đã giấu mặt. Trong khi cơn bão gào thét phía trên tôi, và không có chỗ ẩn nấp. 'Giữa tiếng sấm sét, Chúa tể quý giá, hãy nghe tiếng kêu của tôi. ... |
| 8 | 2544733 | 0 | 0.5031 | vi | Khiến thế giới biến mất Jim Reeves. Anh có nhớ khi yêu em không. Trước khi thế giới đưa tôi lạc lối. Nếu bạn làm vậy thì hãy tha thứ cho tôi. Và làm cho thế giới biến mất. Làm c... |
| 9 | 8229203 | 0 | 0.5021 | vi | Họ gọi gió là Maria. They Call the Wind Maria là một bài hát nổi tiếng của Mỹ với phần lời được viết bởi Alan J. Lerner và phần nhạc của Frederick Loewe cho vở nhạc kịch Broadwa... |
| 10 | 5169842 | 0 | 0.5010 | vi | Bài hát đó là gì? |

Mixed top-10

| rank | docid | rel | retrieval_score_raw | lang | snippet |
|---:|---|---:|---:|---|---|
| 1 | 7928929 | 1 | 0.5914 | vi | Giới thiệu về ÃƒÂ ¢ Ã‚â‚¬Ã‚Å “Take the World By StormÃƒÂ ¢ Ã‚â‚¬Ã‚. ÃƒÂ ¢ Ã‚â‚¬Ã‚Å “Take the World của StormÃƒÂ ¢ Ã‚â‚¬Ã‚ là một bài hát của ban nhạc soul-pop Đan Mạch Lukas Gra... |
| 2 | 1681605 | 0 | 0.5426 | vi | Lưu trữ web với TotalChoice Nhấp và bắt đầu kiếm tiền! Chính sách bảo mật. ĐẾN BÃO BÃO BẰNG (Mosie Lister) Người ghi: Bill Gaither; Vestal Goodman; Tầm nhìn xa hơn; Nguồn cảm hứ... |
| 3 | 2544736 | 0 | 0.5385 | vi | Làm cho thế giới đi xa. Từ Wikipedia, bách khoa toàn thư miễn phí. Make the World Go Away 'là một ca khúc nhạc đồng quê nổi tiếng do Hank Cochran sáng tác. Nó đã trở thành Top 4... |
| 4 | 2544733 | 0 | 0.5369 | vi | Khiến thế giới biến mất Jim Reeves. Anh có nhớ khi yêu em không. Trước khi thế giới đưa tôi lạc lối. Nếu bạn làm vậy thì hãy tha thứ cho tôi. Và làm cho thế giới biến mất. Làm c... |
| 5 | 3659835 | 0 | 0.5293 | vi | Bài hát có những lời này là gì: vỗ tay, vỗ tay, vỗ tay, vỗ tay.? |
| 6 | 5169842 | 0 | 0.5286 | vi | Bài hát đó là gì? |
| 7 | 2672795 | 0 | 0.5265 | vi | Lời dẫn đó và câu chuyện khô khan của tạp chí tin tức đã truyền cảm hứng cho Gordon Lightfoot viết nên một trong những bài hát về câu chuyện hay nhất từ ​​trước đến nay. Vào ngà... |
| 8 | 8229203 | 0 | 0.5253 | vi | Họ gọi gió là Maria. They Call the Wind Maria là một bài hát nổi tiếng của Mỹ với phần lời được viết bởi Alan J. Lerner và phần nhạc của Frederick Loewe cho vở nhạc kịch Broadwa... |
| 9 | 2246224 | 0 | 0.5232 | vi | Họ gọi gió là Maria. They Call the Wind Maria là một bài hát nổi tiếng của Mỹ với phần lời được viết bởi Alan J. Lerner và phần nhạc của Frederick Loewe cho vở nhạc kịch Broadwa... |
| 10 | 3763141 | 0 | 0.5232 | vi | Ai hát bài đó / bài hát đó tên là gì? Cảm ơn đã sử dụng. |

