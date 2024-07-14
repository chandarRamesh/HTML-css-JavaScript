document.getElementById("btn").addEventListener("click",()=>{
    let option=document.getElementById("oper").value;
    // console.log(option);
    let num1=Number(document.getElementById("num1").value);
    console.log(typeof num1);
    let num2=Number(document.getElementById("num2").value);
    console.log(num2);

    let result=document.getElementById("strong");

    switch (option)
    {
        case "add":
            result.innerHTML=(num1+num2);
            break;
        case "sub":
            result.innerHTML=(num1-num2);
            break;
        case "mul":
            result.innerHTML=(num1*num2);
            break;
        case "div":
            result.innerHTML=(num1/num2); 
            break
        default:
            break;       

    }
})