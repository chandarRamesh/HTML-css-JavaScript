const EM=document.querySelector(".emoji");

const earr=[
    "😆",
    "😅",
    "🤣",
    "😂",
    "😀",
    "🤑",
    "🤨",
    "🙂",
    "😊",
    "😗",
    "😛",
    "😏",
    "🤥",
    "😴",
    "🥺",
    "😧",
    "😇",
    "😳",
    "🙃",
    "🥴",
    "🧐",
    "🤨",
    "😒",
    "🤔",
    "🤭",
    "🥰",
    "🤐",
    "😄",
    "🤔",
    "🤪",
    "🥲",
    "😃",
    "😁",
    "😬",
];

EM.addEventListener("mouseover",()=>{
    EM.innerText=earr[Math.floor(Math.random()*earr.length)];
})