var config = {
    type: 'line',
    data: {
        labels: [],
        datasets: []
    },
    options: {
        animation: {
            duration: 0 // disable animation
        },
        elements: {
            line: {
                tension: 0.5 // bezier curves setting
            },
            point:{
                radius: 0 // disable points on line
            }
        },
        responsive: true,
        tooltips: {
            mode: 'index',
            intersect: false,
            position: 'nearest',
        },
        legend: {
            display: true,
            position: 'bottom',
            labels: {
                boxWidth: 35,
                fontSize: 16,
                fontColor: '#F0F0F0'
            }
        },
        hover: {
            mode: 'nearest',
            intersect: true
        },
        scales: {
            scaleLabel: [{fontSize:20}],
            xAxes: [{
                ticks:{
                    maxTicksLimit: 16,
                    fontSize:16,
                    fontColor: '#F0F0F0',
                    min: '6:00',
                    max: '22:00'
                },
                gridLines: {
                  display : false
                }
            }],  
            yAxes: [{
                id: 'A',
                position: 'left',
                display: true,
                gridLines: {
                  drawBorder: false,
                  zeroLineColor: 'rgba(240,240,240,0.1)',
                  color: 'rgba(240,240,240,0.05)'
                },
                ticks: {
                    beginAtZero: true,
                    steps: 10,
                    stepValue: 30,
                    max: 350,
                    fontSize:16,
                    fontColor: '#F0F0F0'
                }
                },{
                id: 'B',
                position: 'right',
                display: true,
                ticks: {
                    beginAtZero: true,
                    steps: 7,
                    stepValue: 1,
                    max: 7,
                    fontSize:16,
                    fontColor: '#F0F0F0'
                },
                gridLines: {
                  drawBorder: false,
                  zeroLineColor: 'transparent',
                  color: 'rgba(240,240,240,0.05)'
                },
                }]
        }
    }
};

const algorithm = {
    AVG: 'Monthly Average',
    EXTRA: 'Extra Trees Regressor',
    LSTM: 'LSTM'
}

const pool_data = {
    PEOPLE: 'Attendance',
    LINES: 'Reserved lines'
}

const chart_color = {}
chart_color[algorithm.AVG] = 'rgba(46,196,182 ,0.9)';
chart_color[algorithm.EXTRA] = 'rgba(255,159,28,0.9)';
chart_color[algorithm.LSTM] = 'rgba(14,124,123,0.9)';
chart_color[pool_data.PEOPLE] = 'rgba(231,29,54,0.7)';
chart_color[pool_data.LINES] = 'rgba(0,0,255,0)';

const month_names = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dec"
}

var selected_date = '2020-01-01';

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
            var day = date.getDate().toString().padStart(2, '0');
            var month = (date.getMonth() + 1).toString().padStart(2, '0');
            var year = date.getFullYear();
            return `${year}-${month}-${day}`;
        }
    });

$( "#datepicker" ).change(function() {
    console.log("datepicker change to "+document.getElementById('datepicker').value);
    var today = picker.toString('YYYY-MM-DD');
    document.getElementById('day_id').innerHTML = today.split(8);
    document.getElementById('month_id').innerHTML = month_names[Number(today.split(5,7))];
    document.getElementById('year_id').innerHTML = today.split(0,4);

    document.getElementById('mse').innerHTML = '';
    updateChart(today);
});

$(window).on('load', function() {
    var d = new Date();
    var month = String(d.getMonth()+1).padStart(2, '0');
    var day = d.getDate().toString().padStart(2, '0');
    var today = d.getFullYear()+'-'+month+'-'+day;
    document.getElementById('day_id').innerHTML = day;
    document.getElementById('month_id').innerHTML = month_names[d.getMonth()+1];
    document.getElementById('year_id').innerHTML = d.getFullYear();
    selected_date = today;
    var ctx = document.getElementById("today").getContext("2d");
    window.myLine = new Chart(ctx, config);
    update_chart(today, true);
});

document.onkeydown = function(event) {
    switch (event.keyCode) {
        case 37:
            left_arrow_date_on_click();
            break;
        case 39:
            right_arrow_date_on_click();
            break;
    }
};

function change_data(data, name)
{
    for(var i = 0; i < window.myLine.config.data.datasets.length; i++)
    {
        if(window.myLine.config.data.datasets[i].label == name)
        {
            window.myLine.config.data.datasets[i].data = data;
        }
    }
}

function compute_mse(algorithm_name, prediction){
    total = 0;
    n = 0;
    for(var i = 72; i < 264; i++){
        if(!isNaN(real_attendance[i]) && !isNaN(prediction[i]))
        {
            total += Math.pow(Number(real_attendance[i]) - Number(prediction[i]), 2);
            n += 1;
        }
    }
    rmse = Math.pow(Number(total/n),0.5).toFixed(0).toString();
    if(rmse=='NaN'){
        rmse = '--'
    }
    switch(algorithm_name){
        case algorithm.AVG:
            document.getElementById('mse_avg').innerHTML = 'RMSE: '+rmse;
            break;
        case algorithm.EXTRA:
            document.getElementById('mse_extra').innerHTML = 'RMSE: '+rmse;
            break;
        case algorithm.LSTM:
            document.getElementById('mse_lstm').innerHTML = 'RMSE: '+rmse;
            break;   
        default:
            console.log('Unknown algorithm: '+algorithm_name);
    }
}

function clear_chart()
{
    for(var i = 0; i < window.myLine.config.data.datasets.length; i++)
    {
        window.myLine.config.data.datasets[i].data = [];
    }
    window.myLine.update();
}

function generate_prediction(data_path, algorithm, generate_new){
    $.ajax({
        type: "GET",  
        url: data_path,
        contentType: "application/json",
        dataType: 'json',      
        success: function(response)  
        {
            str_prediction = response.prediction.split(',');
            prediction = [];
            for (var i = 0; i < str_prediction.length; i++){
                prediction.push(parseInt(str_prediction[i]));
            }
            if(generate_new){
                addData(window.myLine, prediction, true, algorithm);
            }
            else
            {
                change_data(prediction, algorithm);
            }
            compute_mse(algorithm, prediction);
        }   
    });   
}

function add_data(data, data_type) {
    if(data_type == pool_data.LINES){
        window.myLine.data.datasets.push({
            label: "Reserved lines",
            fill: true,
            backgroundColor: "rgba(5,107,191,0.25)",
            borderColor: "rgba(5,107,191,0.35)",
            data: data,
            yAxisID: 'B',
            steppedLine: true,
            borderWidth: 1,
            radius: 0,
        });
    }else{
        window.myLine.data.datasets.push({
            label: data_type,
            backgroundColor: chart_color[data_type],
            borderColor: chart_color[data_type],
            data: data,
            fill: false,
            borderWidth: 3,
            yAxisID: 'A'});
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

function right_arrow_date_on_click(){
    set_date_with_offset(1);
}

function left_arrow_date_on_click(){
    set_date_with_offset(-1);
}

function set_date_with_offset(offset){
    var new_date = new Date(selected_date);
    new_date.setDate(new_date.getDate() + offset);
    var year = String(new_date.getFullYear());
    var month = String(new_date.getMonth()+1).padStart(2, '0');
    var day = new_date.getDate().toString().padStart(2, '0');
    document.getElementById('day_id').innerHTML = day;
    document.getElementById('month_id').innerHTML = month_names[Number(month)];
    document.getElementById('year_id').innerHTML = new_date.getFullYear();
    selected_date = year+'-'+month+'-'+day;
    update_chart(selected_date);
}

function update_chart(date_string, generate_new = false){
    var day_api_string = date_string.split('-').join('/');
    var csv_url = '/get_all_for/'+day_api_string;

    $.ajax({
        type: "GET",  
        url: csv_url,
        contentType: "application/json",
        dataType: 'json',      
        success: function(response)  
        {
            prediction_avg = response.prediction.monthly_average.split(',');
            prediction_extra = response.prediction.extra_trees.split(',');
            real_attendance = response.attendance.split(',');
            if(generate_new){
                window.myLine.data.labels = generate_time_array(date_string);
                add_data(response.attendance.split(','), pool_data.PEOPLE);
                add_data(response.lines_reserved.split(','), pool_data.LINES);
                add_data(prediction_avg, algorithm.AVG);
                add_data(prediction_extra, algorithm.EXTRA);
            }
            else
            {
                change_data(real_attendance, pool_data.PEOPLE);
                change_data(response.lines_reserved.split(','), pool_data.LINES);
                change_data(prediction_avg, algorithm.AVG);
                change_data(prediction_extra, algorithm.EXTRA);
            }
            
            compute_mse(algorithm.AVG, prediction_avg);
            compute_mse(algorithm.EXTRA, prediction_extra);
            window.myLine.update();
        }   
    });
}