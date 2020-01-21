var config = {
    type: 'line',
    data: {
        labels: [],
        datasets: []
    },
    options: {
        responsive: true,
        tooltips: {
            mode: 'index',
            intersect: false,
            position: 'nearest',
        },
        legend: {
            display: true,
            position: 'top',
            labels: {
                boxWidth: 35,
                fontSize: 15,
                fontStyle: 'bold'
            }
        },
        hover: {
            mode: 'nearest',
            intersect: true
        },
        scales: {
            xAxes: [{
                ticks:{
                    maxTicksLimit: 32,
            }}],  
            yAxes: [{
                id: 'A',
                position: 'left',
                display: true,
                ticks: {
                    beginAtZero: true,
                    steps: 10,
                    stepValue: 30,
                    max: 350,
                }
                },{
                id: 'B',
                position: 'right',
                display: true,
                ticks: {
                    beginAtZero: true,
                    steps: 7,
                    stepValue: 1,
                    max: 7
                }
                }]
        }
    }
};

const algorithm = {
    AVG: 'Monthly Average',
    EXTRA: 'Extra Trees Regressor',
    LSTM: 'LSTM'
}

const server_address = 'https://lukoucky.com:5000'

var real_attendance = [];

var config_copy = JSON.parse(JSON.stringify(config));
var today_plus_week = new Date();
today_plus_week.setDate(today_plus_week.getDate() + 7)
var picker = new Pikaday(
    {
        field: document.getElementById('datepicker'),
        bound: false,
        container: document.getElementById('calendar-container'),
        firstDay: 1,
        minDate: new Date(2017, 10, 15),
        maxDate: today_plus_week,
        yearRange: [2017,2020],
        toString(date, format) {
            var day = date.getDate().toString().padStart(2, '0');;
            var month = (date.getMonth() + 1).toString().padStart(2, '0');;
            var year = date.getFullYear();
            return `${year}-${month}-${day}`;
        }
    });

$( "#datepicker" ).change(function() {
  console.log("datepicker change to "+document.getElementById('datepicker').value);
  var today = picker.toString('YYYY-MM-DD');
  document.getElementById('mse').innerHTML = '';
  resetCanvas();
  updateChart(today);
});

$(window).on('load', function() {
    var d = new Date();
    var month = String(d.getMonth()+1).padStart(2, '0');
    var day = d.getDate().toString().padStart(2, '0');
    var today = d.getFullYear()+'-'+month+'-'+day;
    updateChart(today);
});


function updateChart(date_string){
    var day_api_string = date_string.split('-').join('/');
    var csv_url = server_address+'/attendance/'+day_api_string;

    $.ajax({
        type: "GET",  
        url: csv_url,
        dataType: "jsonp",       
        success: function(response)  
        {
            generateChart(response, config, date_string);
            real_attendance = response.attendance.split(',');

            addDataFromCSV(server_address+'/prediction/average/'+day_api_string, 'rgba(67, 175, 105,0.9)', algorithm.AVG);
            addDataFromCSV(server_address+'/prediction/extra_trees/'+day_api_string, 'rgba(102, 46, 155,0.9)', algorithm.EXTRA);
            // addDataFromCSV('data/prediction_random_forest/'+date_string+'.csv', 'rgba(248, 102, 36,0.9)', 'Random Forest Regressor');
            // addDataFromCSV('data/test/prediction_algo2/2019-11-02.csv', 'rgba(67, 175, 105,0.9)', 'Hidden Markov Model');
            // addDataFromCSV('data/test/prediction_algo2/2019-11-02.csv', 'rgba(14,124,123,0.9)', 'Long Short Term Memory');
        }   
    });
    config.options.scales.xAxes[0].ticks.min = '6:00';
    config.options.scales.xAxes[0].ticks.max = '22:00';
}

function compute_mse(algorithm, prediction){
    total = 0;
    n = 0;
    for(var i = 72; i < 264; i++){
        if(!isNaN(real_attendance[i]) && !isNaN(prediction[i]))
        {
            total += Math.pow(Number(real_attendance[i]) - Number(prediction[i]), 2);
            n += 1;
        }
    }
    document.getElementById('mse').innerHTML += '<li>'+algorithm+': '+Number(total/n).toFixed(0)+'</li>';
}

function resetCanvas(){
  const myNode = document.getElementById("graph-container");
  myNode.innerHTML = '<canvas id="today"></canvas>';
  window.myLine = null;
  config = JSON.parse(JSON.stringify(config_copy));
};


function generateChart(data, conf, date_string){
    addData(conf, data.attendance.split(','), "people", false);
    addData(conf, data.lines_reserved.split(','), "lines", false);

    conf.data.labels = generate_time_array(date_string);

    var ctx = document.getElementById("today").getContext("2d");
    window.myLine = new Chart(ctx, conf);
}

 function addDataFromCSV(data_path, color, algorithm){
    $.ajax({
    type: "GET",  
    url: data_path,
    dataType: "jsonp",       
    success: function(response)  
    {
        prediction = response.prediction.split(',')
        addData(window.myLine, prediction, "people", true, color, algorithm);
        compute_mse(algorithm, prediction);
    }   
    });   
 }

 function addData(chart, data, type, updateNow, color, title) {
    color = color || "rgba(234,53,70,1)"
    title = title || 'Actual attendance'
    if(type == 'people'){
    chart.data.datasets.push({
            label: title,
            backgroundColor: color,
            borderColor: color,
            data: data,
            fill: false,
            yAxisID: 'A'})
    }else{
        chart.data.datasets.push({
            label: "Reserved lines",
            fill: true,
            backgroundColor: "rgba(48,188,237,0.25)",
            borderColor: "rgba(0,0,255,0)",
            data: data,
            yAxisID: 'B',
            steppedLine: true,
            radius: 0,
        })
    }

    if(updateNow)
    {
        chart.update();
    }
}

function generate_time_array(date_string){
    var ids = [];
    for (var hour = 0; hour < 24; hour++){
        for (var minute = 0; minute < 60; minute+=5){
            ids.push(hour+':'+minute.toString().padStart(2, '0'));
        }
    }
    return ids;
}
