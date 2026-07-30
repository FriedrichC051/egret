"""打印当前 ai2thor 版本对应的 Linux64(OpenGL) Unity 构建下载 URL 和 commit。"""
import ai2thor._builds as _b
import ai2thor.build as build

commit = _b.COMMIT_ID
b = build.Build(platform="Linux64", commit_id=commit, include_private_scenes=False)
print("COMMIT", commit)
print("URL", b.url)
