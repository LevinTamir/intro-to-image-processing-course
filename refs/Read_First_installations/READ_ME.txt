Install anaconda from the link:

https://www.anaconda.com/download
__________________________________
Move the containtment of this archive to the D:\

(or change all following directories accordingly with the location of the \computer_vision\ folder
__________________________________
in windows update Path variable with (requires administrator privileges):

D:\computer_vision\model\models-master\research

To do it open settings, find "Edit the system environment variables";
press the button "Environment Variables";
in System variables list find Path, choose it and press edit;
then press New and paste the path to \research\ folder
__________________________________
In Anaconda go to enviroments tab, press import; import the enviroment from file: 

D:\computer_vision\READ_FIRST_installations\Course_env.yaml
__________________________________
run in cmd.exe row by row:

cd /d D:\computer_vision\model\models-master\research
protoc object_detection/protos/*.proto --python_out=.
for /f %i in ('dir /b object_detection\protos\*.proto') do protoc --python_out=. object_detection\protos\%i
__________________________________
Now your environment is ready to work, activate it in Anaconda and launch Spyder =)
