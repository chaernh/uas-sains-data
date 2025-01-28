const fs = require('fs');
const Papa = require('papaparse');
const kmeans = require('kmeans-js');

function performKMeans(data) {
  const km = new kmeans({
    K: 3 // Number of clusters
  });

  km.cluster(data);

  console.log(km.centroids);
	logDataToFile(km.centroids, './outputs/centroids-js.json');
}

function logDataToFile(data, filePath) {
  const dataString = JSON.stringify(data, null, 2); // Ubah data menjadi string JSON dengan indentasi

  fs.writeFile(filePath, dataString, (err) => {
    if (err) {
      console.error('Gagal menulis ke file:', err);
    } else {
      console.log('Data berhasil ditulis ke', filePath, ', terdapat', data.length, 'data.');
    }
  });
}

fs.readFile('./dataset_alfa.csv', { encoding: 'utf8' }, (err, csvData) => {
  if (err) {
    console.error("Failed to read file:", err);
    return;
  }

  Papa.parse(csvData, {
    header: true,
    dynamicTyping: true,
    complete: function (results) {
      const data = results.data
      .map(row => [row.X, row.Y, row.Cluster])

      performKMeans(data);
    }
  });
});