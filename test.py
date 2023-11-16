import folium

# 创建地图对象，设置中心点和缩放级别
mymap = folium.Map(location=[25.042264, 121.513047], zoom_start=12)

# 在地图上添加标记
folium.Marker([25.042264, 121.513047], popup='Marker 1').add_to(mymap)
folium.Marker([25.032264, 121.503047], popup='Marker 2').add_to(mymap)

# 将地图保存为 HTML 文件
mymap.save("mymap.html")
