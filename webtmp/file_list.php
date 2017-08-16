<?php

function eachfile($path){
    $files=[];
    foreach(scandir($path) as $v){
        if($v=="."||$v==".."){
            continue;
        }
        $tpath=$path.$v;
        if(is_dir($tpath)){
            $files=array_merge($files,eachfile($tpath.'/'));
        }else{
            $files[]=$path.$v;
        }
    }
    return $files;
}
$files=eachfile('./');
print_r($files) 

?>