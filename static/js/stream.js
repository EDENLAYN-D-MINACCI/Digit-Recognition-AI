window.addEventListener("load", () => {
       

    const buttons = document.getElementsByClassName("filter")

    for(i = 0; i < buttons.length; i++){
      
        buttons[i].addEventListener('click', function(){

            console.log("value:", this.value)

            fetch('http://127.0.0.1:5000/stream', {
                method: 'POST', 
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(this.value)
            })
            .then(response => {
                if(response.ok) console.log("SUCCESS", this.value);             
                else console.log("ERROR", response)
            })  
            .catch(function(error) {
                console.log("Fetch error: " + error)
            });
        })
    }
})