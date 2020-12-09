window.addEventListener("load", () => {


    function setBackgroundColor(){
        context.fillStyle = backgroundColor;
        context.fillRect(0, 0, canvas.width, canvas.height);
    }

    // prediction elements
    var softmax = document.getElementById('softmax')
    var argmax = document.getElementById('argmax')

    // canvas
    const analyzeButton = document.getElementById("analyze-button");
    const clearButton   = document.getElementById("clear-button");
    const canvas        = document.getElementById("canvas");
    const context       = canvas.getContext("2d");

    // canvas parameter
    canvas.width  = window.innerWidth /3
    canvas.height = window.innerWidth /3    
    backgroundColor = "black"
    setBackgroundColor()

    // calculate the canvas offsets
    var BB = canvas.getBoundingClientRect(); 
    var offsetX = BB.left;
    var offsetY = BB.top;       
        
    // line parameter
    var drawing = false;
    context.strokeStyle = "white";
    context.lineWidth = 25;
    context.lineCap = "round";

    // events
    analyzeButton.addEventListener('click', analyzeCanvas)
    clearButton.addEventListener('click', clearCanvas)
    canvas.addEventListener("mousedown", startDrawing);
    canvas.addEventListener("mousemove", draw); // call whenever the mouse is moving
    canvas.addEventListener("mouseup", stopDrawing);



    // canvas functions
    function analyzeCanvas(){
        console.log(canvas.toDataURL());

        fetch('http://127.0.0.1:5000/canvas', {
            method: 'POST', 
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(canvas.toDataURL())
        })
        .then(response => {
            if(response.ok){
                response.json().then((data) => {

                    // showing predictions
                    softmax.innerHTML = data.softmax
                    argmax.innerHTML = data.argmax
                }) 
            } 
            else console.log("ERROR");

        })  
        .catch(function(error) {
            console.log("Fetch error: " + error);
        });
    }

    function clearCanvas(){
        context.clearRect(0, 0, canvas.width, canvas.height);
        setBackgroundColor()
        softmax.innerHTML = ""
        argmax.innerHTML = ""
    }

   


    // drawing functions
    function startDrawing(event){
        context.beginPath();
        context.moveTo(event.clientX - offsetX, event.clientY - offsetY);
        drawing = true;
        draw(event);
    }

    function draw(event){

        if(drawing){       
            context.lineTo(event.clientX - offsetX, event.clientY - offsetY);
            context.stroke();
        }
    }

    function stopDrawing(){
        drawing = false;
    }
})

