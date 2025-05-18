
## 1、用以下代码无法直接加载fbx文件，因为这个操作是针对usd文件的：
```
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = script_dir + '\\noitom\\axis\\\mocap\\avatar\\PN_Stickman_v12_ThumbInward.fbx'
print(model_path)    
omni.kit.commands.execute('CreatePayload',
	usd_context=omni.usd.get_context(),
	path_to='/World/PN_Stickman_v12_ThumbInward',
	asset_path=model_path,
	instanceable=False)
```

## 2、在IsaacSim中可以直接打开fbx文件
打开IsaacSim，在下面的Content标签页浏览到fbx文件，双击即可打开，但是一定要注意，原来的PN_Stickman_v12_ThumbInward.fbx文件是只读的，双击无法打开，需要去掉只读3属性再双击打开。

## 3、通过“另存为”可以存为usd文件，通过这种方式把fbx转为usd
方法1：File->Save As，输入文件名，文件后缀为*.usd，保存即可。
方法2：File->Export 导出usd文件

## 4、程序中加载模型
```
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = script_dir + '\\noitom\\axis\\\mocap\\avatar\\PN_Stickman_v12_ThumbInward.usd'
print(model_path)    
omni.kit.commands.execute('CreatePayload',
	usd_context=omni.usd.get_context(),
	path_to='/World/PN_Stickman_v12_ThumbInward',
	asset_path=model_path,
	instanceable=False)
```
