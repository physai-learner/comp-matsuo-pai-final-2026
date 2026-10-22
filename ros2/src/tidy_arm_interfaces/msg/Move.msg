# 計画の1件。どれを（item）どこへ（place_target）運ぶか。
# 画素は真上画像の上の位置（左上が原点・u が右・v が下）、点は机上のワールド座標 [m]（z は把持高さ）
string item
string place_target
float64 pick_u
float64 pick_v
float64 place_u
float64 place_v
geometry_msgs/Point pick
geometry_msgs/Point place
