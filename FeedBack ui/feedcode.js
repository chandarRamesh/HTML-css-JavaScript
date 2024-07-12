const rating=document.querySelectorAll(".rating");
console.log(rating);
// console.log(ratingsContainer);
const sendbtn=document.querySelector(".btn");
const panel=document.querySelector(".panel-container")

// console.log(sendbtn);
// console.log(panel);

let result="Satisfied";

rating.forEach((x)=>{
    x.addEventListener("click",()=>{
        console.log(x);
        removeactive();

        x.classList.add("active");
         result=x.innerText;
         console.log(result);
    })


})

function removeactive(){
    for (let i = 0; i < rating.length; i++) {
        rating[i].classList.remove("active");
        
    }
}

sendbtn.addEventListener("click",()=>{
    panel.innerHTML=(`<strong class="heart">Thanks for your Feedback!!!</strong>
        <br>
        <p>Feedback:${result}`)
})