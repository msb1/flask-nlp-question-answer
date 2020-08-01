var csocket;
var tsocket;
var plaintext = '<p><br></p>';

$(document).ready(function () {

  csocket = io.connect('http://' + document.domain + ':' + location.port + '/comm');
  csocket.on('message', function (msg) {

    if (msg.type === 'foo') {
    }
    else {
      console.log(msg);
    }
  });

});


$(document).ready(function () {
  $("#menu-toggle").click(function (e) {
    e.preventDefault();
    $("#wrapper").toggleClass("toggled");
  });
});



$(document).ready(function () {

  tsocket = io.connect('http://' + document.domain + ':' + location.port + '/text');
  tsocket.on('message', function (msg) {
    if (msg.type == 'text') {
      plaintext = String(msg.payload);
      // console.log(plaintext);
      edit();
      $('.click2edit').summernote('code', plaintext);
      hideTables();
    }
    else if (msg.type == 'token') {
      // console.log("tokens: " + msg.payload);
      $('#tokenTable').show();
      $('#tokenTable').bootstrapTable({
        data: JSON.parse(msg.payload)
      });
    }
    else if (msg.type == 'noun') {
      // console.log("tokens: " + msg.payload);
      $('#nounTable').show();
      $('#nounTable').bootstrapTable({
        data: JSON.parse(msg.payload)
      });
    }
    else if (msg.type == 'entity') {
      // console.log("tokens: " + msg.payload);
      $('#entityTable').show();
      $('#entityTable').bootstrapTable({
        data: JSON.parse(msg.payload)
      });
    }
    else if (msg.type == 'match') {
      console.log("tokens: " + msg.payload);
      $("#matchDisplay").html(msg.payload);
      $('#matchDisplay').show();
    }
    else if (msg.type == 'sentiment') {
      console.log("tokens: " + msg.payload);
      $("#sentimentDisplay").html(msg.payload);
      $('#sentimentDisplay').show();
    }
    else if (msg.type == 'similarity') {
      console.log("tokens: " + msg.payload);
      $("#similarityDisplay").html(msg.payload);
      $('#similarityDisplay').show();
    }
    else if (msg.type == 'entityDisplay') {
      // console.log("entityDisplay: " + msg.payload);
      $("#entityDisplay").html(msg.payload);
      $('#entityDisplay').show();
    }

  });

});


$(document).ready(function () {
  $('#summernote').summernote({
    buttons: {
      save: SaveButton,
    }
  });
});

// function getText() {
//   var inFile = document.getElementById('inFile').value;
//   var fpath = document.getElementById('fpath').value;
//   var obj = JSON.stringify({ msgType: 'getFileText', 'inFile': inFile, 'filepath': fpath });
//   csocket.emit('message', obj);
// }

function saveText() {
  // var plaintext = $('#summernote').summernote('code').replace(/<[^>]+>/g, '');
  // var plaintext = $("<div>").html($('#summernote').summernote('code')).text().trim();
  plaintext = $("<div>").html($('.click2edit').summernote('code')).text().trim().replace(/\n/g, '');
  console.log(plaintext);
  $('.click2edit').summernote('destroy');
  var outFile = document.getElementById('outFile').value;
  var fpath = document.getElementById('fpath').value;
  var obj = JSON.stringify({ msgType: 'saveText', 'outFile': outFile, 'filepath': fpath, 'plaintext': plaintext });
  tsocket.emit('message', obj);
};

var SaveButton = function (context) {
  var ui = $.summernote.ui;
  var button = ui.button({
    contents: '<i class="fa fa-save"/>',
    click: function () {
      saveText();
    }
  });
  return button.render();
};

function match() {

  var p1 = document.getElementById('phrase1').value;
  var p2 = document.getElementById('phrase2').value;
  var p3 = document.getElementById('phrase3').value;
  var obj = JSON.stringify({ msgType: 'matcher', phrase1: p1, phrase2: p2, phrase3: p3 });
  csocket.emit('message', obj);
}

function similarity() {

  var c1 = document.getElementById('compareText').value;
  var obj = JSON.stringify({ msgType: 'similarity', compareText: c1 });
  csocket.emit('message', obj);
}

function sentiment() {

  var obj = JSON.stringify({ msgType: 'sentiment', exec: 'True' });
  csocket.emit('message', obj);
}

function hideTables() {
  $('#tokenTable').hide();
  $('#nounTable').hide();
  $('#entityTable').hide();
  $('#entityDisplay').hide();
  $('#matchDisplay').hide();
  $('#sentimentDisplay').hide();
  $('#similarityDisplay').hide();
}

