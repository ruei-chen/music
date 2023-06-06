

function initMap() {
  console.log(google); // 確認 google 物件是否已經定義
  console.log(google.maps); // 確認 google.maps 物件是否已經定義
  // 創建一個新的地圖對象
  var map = new google.maps.Map(document.getElementById('map'), {
    zoom: 8, // 地圖縮放級別
    center: { lat: 25.042029, lng: 121.534846 } // 中心座標
  });
  var searchService = new google.maps.places.PlacesService(map);

  document.getElementById('locate-btn').addEventListener('click', function() {
    if (navigator.geolocation) {
      // 獲取地理位置
      navigator.geolocation.getCurrentPosition(successCallback, errorCallback);
    } else {
      // 瀏覽器不支援地理位置API
      alert('瀏覽器不支援地理位置功能');
    }
  });

  function successCallback(position) {
    var latitude = position.coords.latitude; // 緯度
    var longitude = position.coords.longitude; // 經度

    // 將緯度和經度填充到表單中
    document.getElementById('latitude').value = latitude;
    document.getElementById('longitude').value = longitude;

    // 創建使用者的LatLng對象
    var userLocation = new google.maps.LatLng(latitude, longitude);

    // 在地圖上標記使用者的位置
    var marker = new google.maps.Marker({
      position: userLocation,
      map: map
    });

    // 將地圖中心設置為使用者的位置
    map.setCenter(userLocation);
  }

  function errorCallback(error) {
    switch (error.code) {
      case error.PERMISSION_DENIED:
        alert("使用者拒絕提供地理位置");
        break;
      case error.POSITION_UNAVAILABLE:
        alert("無法獲取地理位置資訊");
        break;
      case error.TIMEOUT:
        alert("獲取地理位置超時");
        break;
      case error.UNKNOWN_ERROR:
        alert("發生未知錯誤");
        break;
    }
  }

  document.getElementById('search-btn').addEventListener('click', function() {
    var lat = parseFloat(document.getElementById('latitude').value);
    var lng = parseFloat(document.getElementById('longitude').value);
    var location = new google.maps.LatLng(lat, lng);

    var request = {
      location: location,
      radius: 1000000, // 搜尋半徑，單位為公尺
      query: lat + ',' + lng,
      fields: ['name', 'geometry']
    };

    searchService.textSearch(request, function(results, status) {
      if (status === google.maps.places.PlacesServiceStatus.OK) {
        // 成功搜尋到地點，將地圖移動到該地點位置
        var place = results[0];
        map.setCenter(place.geometry.location);
        // 在地圖上標記出該地點
        var marker = new google.maps.Marker({
          map: map,
          position: place.geometry.location
        });
      } else {
        // 搜尋失敗，提示使用者
        alert('很抱歉，找不到符合搜尋條件的地點。');
      }
    });
  });
}

window.initMap = initMap;