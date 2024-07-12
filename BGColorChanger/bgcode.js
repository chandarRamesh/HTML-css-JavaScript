const btns=document.querySelectorAll(".btn");
console.log(btns);
const bd=document.querySelector("body");
// console.log(bd);


btns.forEach((x)=>{
    x.addEventListener("click",()=>{
        bd.classList="";
        console.log(x);
        let color=x.value;
        console.log(color);
        changeBackground(color);

    })
})

function changeBackground(color){
    switch (color)
    {
        case "purple":
            bd.classList.add("purple");
            break;

        case "blue":
            bd.classList.add("blue");
            break;

        case "red":
            bd.classList.add("red");
            break;
        
            case "green":
                bd.classList.add("green");
                break;

            
        case "yellow":
            bd.classList.add("yellow");
            break;

        
            case "teal":
                bd.classList.add("teal");
                break;

            
        case "BG":
            bd.classList.add("BG");
            break;
        
            default:
                break;

    }

}